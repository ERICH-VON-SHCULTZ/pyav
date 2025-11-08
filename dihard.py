import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from pytorch_lightning.loggers import WandbLogger

# Deal with input arguments

parser = argparse.ArgumentParser(description ='Performs diarization using pyannote dihard recipe')

parser.add_argument("--system", default="pyannote/speaker-diarization-3.1", dest="system")
parser.add_argument("--pretrainedp", default="pyannote/speaker-diarization-3.1", dest="pretrainedp")
parser.add_argument("--pretraineds", default="pyannote/segmentation-3.0", dest="pretraineds")
parser.add_argument("--groundtruths", default="/scratch/map22-share/metadata/test5.rttm", dest="groundtruths")
parser.add_argument('--thresh', type=float, default=-1.0, dest="thresh")
parser.add_argument('--clusiter', type=int, default=20, dest="clusiter")
parser.add_argument('--clussize', type=int, default=15, dest="clussize")
parser.add_argument('--epochs', type=int, default=20, dest="epochs")
parser.add_argument("--hyper", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--hyperset", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--rttm", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--rttmdir", default=".", dest='rttmdir')
parser.add_argument("--rttmhyp", default="", dest='rttmhyp')
parser.add_argument("--project", default="dihard", dest='project')
parser.add_argument("--name", default="test", dest='name')
parser.add_argument("--finetune", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--justaudio", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--justvideo", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--cdir", default=".", dest="cdir")
parser.add_argument("--data", default="test", dest="data")
parser.add_argument("--speakers", type=int, default=0, dest="speakers")
parser.add_argument("--nometric", default=False, dest="nometric")
parser.add_argument("--collar", type=float, default=0.0, dest="collar")
parser.add_argument("--model", type=int, default=0, dest="model")
parser.add_argument("--esize", type=int, default=1024, dest="esize")
parser.add_argument("--protocol", default='headcam16.SpeakerDiarization.try1', dest="protocol")

args = parser.parse_args()
diarization_system = args.system
hyper = args.hyper
hyperset = args.hyperset
pretrained_diarization_system = args.pretrainedp
pretrained_segmentation_system = args.pretraineds
clusiter = args.clusiter
clussize = args.clussize
finetune = args.finetune
cdir     = args.cdir
thresh   = args.thresh
groundtruths = args.groundtruths
data     = args.data
speakers = args.speakers
collar   = args.collar
epochs   = args.epochs
project  = args.project
name     = args.name
justaudio  = args.justaudio
justvideo  = args.justvideo
model      = args.model
esize      = args.esize

if hyperset:
    print("Cluster size is: ", clussize, " thresh is ", thresh)
    

# Get input processing setup (from pyannote)
from pyannote.database import get_protocol
from pyannote.database import FileFinder
preprocessors = {'audio': FileFinder()}
protocol = get_protocol(args.protocol, preprocessors=preprocessors)

if data == "test" or data == "rttm":
    data_protocol = protocol.test()
elif data == "dev":
    data_protocol = protocol.development()
elif data == "train":
    data_protocol = protocol.train()
else:
    print("No such protocol: " + data)
    exit()

from pathlib import Path
from io import IOBase
from typing import Mapping, Optional, Text, Tuple, Union
from torch import Tensor
from pyannote.core import Segment
from pyannote.audio.core.io import Npy
from pyannote.audio.core.model import Model
from pyannote.audio.models.segmentation import PyanNet
import torch.nn.functional as F
from einops import rearrange

class CoAttentionEncoder(nn.Module):
    """Co-attention encoder for multimodal fusion, returns enhanced embeddings"""
    def __init__(self, audio_embed_dim=128, video_embed_dim=60, num_heads=8, num_layers=2, 
                 dim_feedforward=512, dropout=0.1, activation='relu'):
        super().__init__()
        self._num_layers = num_layers
        self.audio_embed_dim = audio_embed_dim
        self.video_embed_dim = video_embed_dim
        
        # Audio self-attention layers
        self.audio_self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=audio_embed_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Video self-attention layers
        video_heads = max(1, num_heads // 2)  # Ensure divisibility
        assert video_embed_dim % video_heads == 0, f"video_embed_dim ({video_embed_dim}) must be divisible by video_heads ({video_heads})"
        
        self.video_self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=video_embed_dim,
                nhead=video_heads,
                dim_feedforward=dim_feedforward // 2,
                dropout=dropout,
                activation=activation,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Cross-attention layers: Audio queries Video
        self.video_to_audio_proj = nn.Linear(video_embed_dim, audio_embed_dim)
        self.audio_queries_video_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=audio_embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Cross-attention layers: Video queries Audio  
        self.audio_to_video_proj = nn.Linear(audio_embed_dim, video_embed_dim)
        self.video_queries_audio_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=video_embed_dim,
                num_heads=video_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Layer normalization for residual connections
        self.audio_layer_norms = nn.ModuleList([
            nn.LayerNorm(audio_embed_dim) for _ in range(num_layers)
        ])
        self.video_layer_norms = nn.ModuleList([
            nn.LayerNorm(video_embed_dim) for _ in range(num_layers)
        ])

    def forward(self, audio_emb, video_emb, audio_mask=None, video_mask=None):
        """
        Args:
            audio_emb: [batch, frames, audio_embed_dim] - Audio embeddings from upstream
            video_emb: [batch, frames, video_embed_dim] - Video embeddings after interpolation
            
        Returns:
            enhanced_audio: [batch, frames, audio_embed_dim] - Enhanced audio embeddings
            enhanced_video: [batch, frames, video_embed_dim] - Enhanced video embeddings
        """
        for i in range(self._num_layers):
            # Audio self-attention
            audio_emb_self = self.audio_self_attn_layers[i](
                src=audio_emb,
                src_mask=audio_mask
            )
            
            # Video self-attention
            video_emb_self = self.video_self_attn_layers[i](
                src=video_emb,
                src_mask=video_mask
            )
            
            # Cross-attention 1: Audio queries Video
            video_kv = self.video_to_audio_proj(video_emb_self)
            audio_cross_attn, _ = self.audio_queries_video_attn_layers[i](
                query=audio_emb_self,
                key=video_kv,
                value=video_kv
            )
            
            # Cross-attention 2: Video queries Audio
            audio_kv = self.audio_to_video_proj(audio_emb_self)
            video_cross_attn, _ = self.video_queries_audio_attn_layers[i](
                query=video_emb_self,
                key=audio_kv,
                value=audio_kv
            )
            
            # Residual connections with layer normalization
            audio_emb = self.audio_layer_norms[i](audio_emb + audio_cross_attn)
            video_emb = self.video_layer_norms[i](video_emb + video_cross_attn)
        
        return audio_emb, video_emb



# class ModifiedModel(torch.nn.Module):
# class ModifiedModel(Model):
class ModifiedModel(PyanNet):

    @property
    def dimension(self) -> int:
        """Dimension of output"""
        if isinstance(self.specifications, tuple):
            raise ValueError("PyanNet does not support multi-tasking.")

        if self.specifications.powerset:
            return self.specifications.num_powerset_classes
        else:
            return len(self.specifications.classes)

    def __example_input_array(self, duration: Optional[float] = None):
        duration = duration or next(iter(self.specifications)).duration
        fake_audio = torch.randn(
            (
                1,
                self.hparams.num_channels,
                self.audio.get_num_samples(duration),
            ),
            device=self.device,
        )

        fake_npy = torch.randn((1, int(duration * 30), esize,), device=self.device,)

        return [fake_audio, fake_npy] 

    @property
    def example_input_array(self):
        return self.__example_input_array()


    def __init__(self, original_model=None, specifications=None, hparams=None):
#       super(ModifiedModel, self).__init__()
        super().__init__()
        # Keep the original model 
#       self.original_model = original_model

        if original_model != None:
            if specifications == None:
                self.specifications = original_model.specifications
            else:
                self.specifications = specifications
            if hparams == None:
                self.hparams.update(vars(original_model.hparams))
            else:
                self.hparams.update(vars(hparams))
            self.classifier = original_model.classifier 
            self.activation = original_model.activation 
        else:
            if specifications != None:
                self.specifications = specifications
            if hparams != None:
                self.hparams.update(vars(hparams))

#        print("Hparams")
#        print(self.hparams)

                
#        if self.hparams.linear["num_layers"] > 0:
#            in_features = self.hparams.linear["hidden_size"]
#        else:
#            in_features = self.hparams.lstm["hidden_size"] * (
#                2 if self.hparams.lstm["bidirectional"] else 1
#            )

##       self.classifier = torch.nn.Linear(in_features, self.dimension)
            
##       self.hparams = original_model.hparams
##       self.dimension = original_model.dimension
##       self.device = original_model.device

        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                m.weight.data.fill_(1.0)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

        self.npy = Npy()

        if model == 0 or model == 3 or model == 4:
            self.second_input = torch.nn.Sequential(
                torch.nn.Linear(esize, 60),
                torch.nn.ReLU(),
                torch.nn.Linear(60, 60),
                torch.nn.ReLU())
            if model == 0:
                self.merge = torch.nn.Sequential(
                    torch.nn.Linear(120, 60),
                    torch.nn.ReLU())
            elif model == 3:
                self.merge = torch.nn.Sequential(
                    torch.nn.Linear(188, 128),
                    torch.nn.ReLU())
            else:
                self.merge = torch.nn.Sequential(
                    torch.nn.Linear(188, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 128),
                    torch.nn.ReLU())
                
            
        elif model == 1:
            self.second_input = torch.nn.Sequential(
                torch.nn.Linear(esize, 60),
                torch.nn.ReLU(),
                torch.nn.Linear(60, 60),
                torch.nn.ReLU(),
                torch.nn.Linear(60, 60),
                torch.nn.ReLU(),
                torch.nn.Linear(60, 60),
                torch.nn.ReLU())
        
            self.merge = torch.nn.Sequential(
                torch.nn.Linear(120, 60),
                torch.nn.ReLU())
            
        elif model == 2:
            self.second_input = torch.nn.Sequential(
                torch.nn.Linear(esize, 60),
                torch.nn.ReLU(),
                torch.nn.Linear(60, 60),
                torch.nn.ReLU())
        
            self.merge = torch.nn.Sequential(
                torch.nn.Linear(120, 60),
                torch.nn.ReLU(),
                torch.nn.Linear(60, 60),
                torch.nn.ReLU())

        elif model == 6:
            # Model 6: late fusion + CoAttentionEncoder
            self.second_input = torch.nn.Sequential(
                torch.nn.Linear(esize, 60),
                torch.nn.ReLU(),
                torch.nn.Linear(60, 60),
                torch.nn.ReLU())
            
            
            self.co_attention_encoder = CoAttentionEncoder(
                audio_embed_dim=128,  
                video_embed_dim=60,
                num_heads=4,
                num_layers=2,
                dropout=0.1
            )

            self.merge = torch.nn.Sequential(
                torch.nn.Linear(188, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 128),
                torch.nn.ReLU()
            )
            # self.merge = torch.nn.Sequential(
            #     torch.nn.Linear(188, 256),
            #     torch.nn.ReLU(),
            #     torch.nn.Linear(256, 128),
            #     torch.nn.ReLU(),
            #     torch.nn.Linear(128, 128),
            #     torch.nn.ReLU()
            # )

        else:
            print("Model %d not implemented" % model)
            exit()
            

    def prepare_data(self):
        self.task.prepare_data()

    def build(self):

        if self.hparams.linear["num_layers"] > 0:
           in_features = self.hparams.linear["hidden_size"]
        else:
           in_features = self.hparams.lstm["hidden_size"] * (
               2 if self.hparams.lstm["bidirectional"] else 1
           )

        if not hasattr(self, "classifier"):
            self.classifier = torch.nn.Linear(in_features, self.dimension)
        if not hasattr(self, "activation"):
            self.activation = self.default_activation()


    def setup(self, stage=None):
        if stage == "fit":
            # let the task know about the trainer (e.g for broadcasting
            # cache path between multi-GPU training processes).
            self.task.trainer = self.trainer

        # setup the task if defined (only on training and validation stages,
        # but not for basic inference)
        if self.task:
            self.task.setup(stage)

        # list of layers before adding task-dependent layers
        before = set((name, id(module)) for name, module in self.named_modules())

        # add task-dependent layers (e.g. final classification layer)
        # and re-use original weights when compatible

        original_state_dict = self.state_dict()
        self.build()

        try:
            missing_keys, unexpected_keys = self.load_state_dict(
                original_state_dict, strict=False
            )

        except RuntimeError as e:
            if "size mismatch" in str(e):
                msg = (
                    "Model has been trained for a different task. For fine tuning or transfer learning, "
                    "it is recommended to train task-dependent layers for a few epochs "
                    f"before training the whole model: {self.task_dependent}."
                )
                warnings.warn(msg)
            else:
                raise e

        # move layers that were added by build() to same device as the rest of the model
        for name, module in self.named_modules():
            if (name, id(module)) not in before:
                module.to(self.device)

        # add (trainable) loss function (e.g. ArcFace has its own set of trainable weights)
        if self.task:
            # let task know about the model
            self.task.model = self
            # setup custom loss function
            self.task.setup_loss_func()
            # setup custom validation metrics
            self.task.setup_validation_metric()

        # list of layers after adding task-dependent layers
        after = set((name, id(module)) for name, module in self.named_modules())

        # list of task-dependent layers
        self.task_dependent = list(name for name, _ in after - before)

    def forward(self, audio, npy) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)

        Returns
        -------
        scores : (batch, frame, classes)
        """

#        f = open('out.txt', 'a')
#        print("Reached forward pass", file=f)
#        print("Audio size", file=f)
#        print(audio.size(), file=f)
#        print("npy size", file=f)
#        print(npy.size(), file=f)
#       print("audio", file=f)
#       print(audio, file=f)

        outputs_sincnet = self.sincnet(audio)

#       for name, param in self.sincnet.named_parameters():
#           print(f"Layer: {name} | Size: {param.size()} | Values: {param}")
#        print("self.lstm")
#        for name, param in self.lstm.named_parameters():
#            print(f"Layer: {name} | Size: {param.size()} | Values: {param}")
#        print("self.linear")
#        for name, param in self.linear.named_parameters():
#            print(f"Layer: {name} | Size: {param.size()} | Values: {param}")
#        print("self.classifier")
#        for name, param in self.classifier.named_parameters():
#            print(f"Layer: {name} | Size: {param.size()} | Values: {param}")
#        print("outputs", file=f)
#        print(outputs_sincnet, file=f)
#        print("outputs_sincnet.shape", file=f)
#        print(outputs_sincnet.shape, file=f)
        
        outputs_npy = self.second_input(npy)

        sincnet_frames = outputs_sincnet.size()[2]

#        print("outputs_npy.shape", file=f)
#        print(outputs_npy.shape, file=f)

        outputs_npy_interp = F.interpolate(rearrange(outputs_npy, "batch feature frame -> batch frame feature"), sincnet_frames)

#        print("outputs_npy_interp.shape", file=f)
#        print(outputs_npy_interp.shape, file=f)

        if not (model == 3 or model == 4 or model == 6):
            concat_outputs = torch.cat((outputs_sincnet, outputs_npy_interp),1)
            outputs = rearrange(self.merge(rearrange(concat_outputs,"batch feature frame -> batch frame feature")), "batch frame feature -> batch feature frame")
        
#
#       For processing audio input only. It ignores the embeddings.
#
        if justaudio and justvideo:
            print("Both justaudio and justvideo are set. Please make up your mind!!!")
        if justaudio or model == 3 or model == 4 or model == 6:
            outputs = outputs_sincnet
        if justvideo:
            outputs = outputs_npy_interp
        
        if self.hparams.lstm["monolithic"]:
            outputs, _ = self.lstm(
                rearrange(outputs, "batch feature frame -> batch frame feature")
            )
        else:
            outputs = rearrange(outputs, "batch feature frame -> batch frame feature")
            for i, lstm in enumerate(self.lstm):
                outputs, _ = lstm(outputs)
                if i + 1 < self.hparams.lstm["num_layers"]:
                    outputs = self.dropout(outputs)

#        print("outputs for lstm", file=f)
#        print(outputs, file=f)
#        print("outputs_lstm.shape", file=f)
#        print(outputs.shape, file=f)
        

        if self.hparams.linear["num_layers"] > 0:
            for linear in self.linear:
                outputs = F.leaky_relu(linear(outputs))

#        print("outputs for linear", file=f)
#        print(outputs, file=f)
#        print("outputs_linear.shape", file=f)
#        print(outputs.shape, file=f)

        if model == 6:
            # outputs: [batch, frames, audio_dim]
            # outputs_npy_interp: [batch, video_dim, frames] 
            
            #  [batch, video_dim, frames] -> [batch, frames, video_dim]
            video_features = rearrange(outputs_npy_interp, "batch feature frame -> batch frame feature")

            outputs, video_features = self.co_attention_encoder(outputs, video_features)
            
            concat_outputs = torch.cat((outputs, video_features), 2)
            outputs = self.merge(concat_outputs)

        if model == 3 or model == 4:
            concat_outputs = torch.cat((outputs, rearrange(outputs_npy_interp, "batch feature frame -> batch frame feature")),2)
            outputs = self.merge(concat_outputs)

        outputs = self.classifier(outputs)

#        print("outputs for classifier", file=f)
#        print(outputs, file=f)
#        print("outputs_classifier.shape", file=f)
#        print(outputs.shape, file=f)
#        f.close()

        return self.activation(outputs)
    


# Finetuning

# Read model/pipeline 

from pyannote.audio import Pipeline, Model

if not finetune:
    pretrained_pipeline = Pipeline.from_pretrained(pretrained_diarization_system, use_auth_token="YourHFtoken")
    base_segmentation_model = Model.from_pretrained(pretrained_segmentation_system, use_auth_token="hf_ypqKCKCUFLMFz zICjGsVmKDBTqbZLhJLZv")
    pretrained_segmentation_model = ModifiedModel(base_segmentation_model)
    # 3. Transfer weights
    # Check if both models have the same keys in their state dictionaries
    base_segmentation_dict = base_segmentation_model.state_dict()
    pretrained_segmentation_dict = pretrained_segmentation_model.state_dict()

    # Filter out unnecessary keys from the pre-trained model
    base_segmentation_dict = {k: v for k, v in base_segmentation_dict.items() if k in pretrained_segmentation_dict}

    # Load the filtered weights into the derived model
    # Freeze parameters
    pretrained_segmentation_dict.update(base_segmentation_dict)
    for name, param in pretrained_segmentation_dict.items():
        param.requires_grad = True
        
    pretrained_segmentation_model.load_state_dict(pretrained_segmentation_dict)

    from pyannote.audio.pipelines import SpeakerDiarization
    pipeline = SpeakerDiarization(
        segmentation=pretrained_segmentation_model,
        segmentation_batch_size=32,
        embedding=pretrained_pipeline.embedding,
        embedding_exclude_overlap=pretrained_pipeline.embedding_exclude_overlap,
        clustering=pretrained_pipeline.klustering)

    pipeline.instantiate({
        "clustering": {
            "threshold": pretrained_pipeline._pipelines['clustering']._instantiated['threshold'],
            "min_cluster_size": pretrained_pipeline._pipelines['clustering']._instantiated['min_cluster_size'],
            "method": pretrained_pipeline._pipelines['clustering']._instantiated['method'],
            },
        "segmentation": {
            "min_duration_off": pretrained_pipeline._pipelines['segmentation']._instantiated['min_duration_off'],
            },
        },
    )

else:

# First optimize segmentation model

    base_segmentation_model = Model.from_pretrained(pretrained_segmentation_system, use_auth_token="hf_ypqKCKCUFLMFz zICjGsVmKDBTqbZLhJLZv")
    print("***************ORIGINAL MODEL****************")
#    from pprint import pprint
#    print("Original Model's state_dict:")
#    for param_tensor in pretrained_segmentation_model.state_dict():
#         print(param_tensor, "\t", pretrained_segmentation_model.state_dict()[param_tensor].size())

#    print("self.sincnet")
#    for name, param in base_segmentation_model.sincnet.named_parameters():
#         print(f"Layer: {name} | Size: {param.size()} | Values: {param}")
#    print("self.lstm")
#    for name, param in base_segmentation_model.lstm.named_parameters():
#        print(f"Layer: {name} | Size: {param.size()} | Values: {param}")
#    print("self.linear")
#    for name, param in base_segmentation_model.linear.named_parameters():
#        print(f"Layer: {name} | Size: {param.size()} | Values: {param}")
#    print("self.classifier")
#    for name, param in base_segmentation_model.classifier.named_parameters():
#            print(f"Layer: {name} | Size: {param.size()} | Values: {param}")

    if not justvideo:
        pretrained_segmentation_model = ModifiedModel(base_segmentation_model)
    else:
        pretrained_segmentation_model = ModifiedModel(specifications=base_segmentation_model.specifications, hparams=base_segmentation_model.hparams)

    print("***************MODIFIED MODEL****************")
#    print(pretrained_segmentation_model)
#    pprint(vars(pretrained_segmentation_model))
#    print("Modified Model's state_dict:")
#    for param_tensor in pretrained_segmentation_model.state_dict():
#         print(param_tensor, "\t", pretrained_segmentation_model.state_dict()[param_tensor].size())
    
    from pyannote.audio.tasks import Segmentation2
    task = Segmentation2(
        protocol, 
        duration=base_segmentation_model.specifications.duration, 
        max_num_speakers=len(base_segmentation_model.specifications.classes), 
        batch_size=32,
        num_workers=1, 
        max_speakers_per_chunk=3,
        max_speakers_per_frame=2,
        loss="bce")
    pretrained_segmentation_model.task = task
#   pretrained_segmentation_model.setup(stage="fit")
    pretrained_segmentation_model.prepare_data()
    pretrained_segmentation_model.setup()

    if not justvideo:
        # Check if both models have the same keys in their state dictionaries
        base_segmentation_dict = base_segmentation_model.state_dict()
        pretrained_segmentation_dict = pretrained_segmentation_model.state_dict()

        # Filter out unnecessary keys from the pre-trained model
        base_segmentation_dict = {k: v for k, v in base_segmentation_dict.items() if k in pretrained_segmentation_dict}

        # Load the filtered weights into the derived model
        pretrained_segmentation_dict.update(base_segmentation_dict)
        pretrained_segmentation_model.load_state_dict(pretrained_segmentation_dict)


#    print("self.sincnet")
#    for name, param in pretrained_segmentation_model.sincnet.named_parameters():
#         print(f"Layer: {name} | Size: {param.size()} | Values: {param}")
#    print("self.lstm")
#    for name, param in pretrained_segmentation_model.lstm.named_parameters():
#        print(f"Layer: {name} | Size: {param.size()} | Values: {param}")
#    print("self.linear")
#    for name, param in pretrained_segmentation_model.linear.named_parameters():
#        print(f"Layer: {name} | Size: {param.size()} | Values: {param}")
#    print("self.classifier")
#    for name, param in pretrained_segmentation_model.classifier.named_parameters():
#            print(f"Layer: {name} | Size: {param.size()} | Values: {param}")

    
#    print("sincnet")
#    for name, param in pretrained_segmentation_model.sincnet.named_parameters():
#       param.requires_grad = False
#    print("lstm")
#    for name, param in pretrained_segmentation_model.lstm.named_parameters():
#       param.requires_grad = False
#    print("linear")
#    for name, param in pretrained_segmentation_model.linear.named_parameters():
#       param.requires_grad = False
#    print("classifier")
#    for name, param in pretrained_segmentation_model.classifier.named_parameters():
#       param.requires_grad = True

#    print("pretrained model")
#    print(pretrained_segmentation_model)
#    print("pretrained model variables")
#    print(vars(pretrained_segmentation_model))
    from types import MethodType
    from torch.optim import Adam
    from pytorch_lightning.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        RichProgressBar,
    )

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-4)

    pretrained_segmentation_model.configure_optimizers = MethodType(configure_optimizers, pretrained_segmentation_model)

    # we monitor diarization error rate on the validation set
    # and use to keep the best checkpoint and stop early
    monitor, direction = task.val_monitor
    checkpoint_path = cdir + "/" + project + "/" + name
    checkpoint = ModelCheckpoint(
        monitor=monitor,
        mode=direction,
        dirpath=checkpoint_path,
        save_top_k=1,
        every_n_epochs=1,
        save_last=False,
        save_weights_only=False,
        filename="{epoch}",
        verbose=True,
    )
    early_stopping = EarlyStopping(
        monitor=monitor,
        mode=direction,
        min_delta=0.0,
        patience=10,
        strict=True,
        verbose=False,
    )

    callbacks = [RichProgressBar(), checkpoint, early_stopping]

    # we train for at most 20 epochs (might be shorter in case of early stopping)
    from pytorch_lightning import Trainer
    wandb_logger = WandbLogger(log_model="all", project=project, name=name, save_dir=cdir)
    
    trainer = Trainer(accelerator="gpu", 
                      default_root_dir=cdir,
                      callbacks=callbacks,
                      strategy="ddp_find_unused_parameters_true",
                      logger=wandb_logger,
                      max_epochs=epochs,
                      devices=1,
                      gradient_clip_val=0.5)

#    print("***********BEFORE TRAINING**************")
#    print("bef.sincnet")
#    for name, param in pretrained_segmentation_model.sincnet.named_parameters():
#        print(f"Layer: {name} | Size: {param.size()} | Values: {param}")
#    print("bef.lstm")
#    for name, param in pretrained_segmentation_model.lstm.named_parameters():
#        print(f"Layer: {name} | Size: {param.size()} | Values: {param}")
#    print("bef.linear")
#    for name, param in pretrained_segmentation_model.linear.named_parameters():
#        print(f"Layer: {name} | Size: {param.size()} | Values: {param}")
#    print("bef.classifier")
#    for name, param in pretrained_segmentation_model.classifier.named_parameters():
#            print(f"Layer: {name} | Size: {param.size()} | Values: {param}")
#    print("bef.second_iput")
#    for name, param in pretrained_segmentation_model.second_input.named_parameters():
#            print(f"Layer: {name} | Size: {param.size()} | Values: {param}")
#    print("bef.merge")
#    for name, param in pretrained_segmentation_model.merge.named_parameters():
#            print(f"Layer: {name} | Size: {param.size()} | Values: {param}")

    
    trainer.fit(pretrained_segmentation_model)

#    print("***********AFTER TRAINING**************")
#    print("aft.sincnet")
#    for name, param in pretrained_segmentation_model.sincnet.named_parameters():
#         print(f"Layer: {name} | Size: {param.size()} | Values: {param}")
#    print("aft.lstm")
#    for name, param in pretrained_segmentation_model.lstm.named_parameters():
#        print(f"Layer: {name} | Size: {param.size()} | Values: {param}")
#    print("aft.linear")
#    for name, param in pretrained_segmentation_model.linear.named_parameters():
#        print(f"Layer: {name} | Size: {param.size()} | Values: {param}")
#    print("aft.classifier")
#    for name, param in pretrained_segmentation_model.classifier.named_parameters():
#            print(f"Layer: {name} | Size: {param.size()} | Values: {param}")
#    print("aft.second_iput")
#    for name, param in pretrained_segmentation_model.second_input.named_parameters():
#            print(f"Layer: {name} | Size: {param.size()} | Values: {param}")
#    print("aft.merge")
#    for name, param in pretrained_segmentation_model.merge.named_parameters():
#            print(f"Layer: {name} | Size: {param.size()} | Values: {param}")


    # save path to the best checkpoint for later use
    finetuned_model_path = checkpoint.best_model_path
#   finetuned_model = ModifiedModel(finetuned_model_path, specifications=pretrained_segmentation_model.specifications, hparams=pretrained_segmentation_model.hparams)
    finetuned_model = ModifiedModel.from_pretrained(finetuned_model_path)
        
#    print("***********AFTER RE-READING**************")
#    print("ft.sincnet")
#    for name, param in finetuned_model.sincnet.named_parameters():
#         print(f"Layer: {name} | Size: {param.size()} | Values: {param}")
#    print("ft.lstm")
#    for name, param in finetuned_model.lstm.named_parameters():
#        print(f"Layer: {name} | Size: {param.size()} | Values: {param}")
#    print("ft.linear")
#    for name, param in finetuned_model.linear.named_parameters():
#        print(f"Layer: {name} | Size: {param.size()} | Values: {param}")
#    print("ft.classifier")
#    for name, param in finetuned_model.classifier.named_parameters():
#            print(f"Layer: {name} | Size: {param.size()} | Values: {param}")
#    print("ft.second_iput")
#    for name, param in finetuned_model.second_input.named_parameters():
#            print(f"Layer: {name} | Size: {param.size()} | Values: {param}")
#    print("ft.merge")
#    for name, param in finetuned_model.merge.named_parameters():
#            print(f"Layer: {name} | Size: {param.size()} | Values: {param}")
#
#
##   hyper = True
#
    from pyannote.audio.pipelines import SpeakerDiarization

    # Initial full pipeline with entries from the pretrained system
    
    pretrained_pipeline = Pipeline.from_pretrained(pretrained_diarization_system, use_auth_token="YourHFtoken")

    pipeline = SpeakerDiarization(
        segmentation=finetuned_model,
        embedding=pretrained_pipeline.embedding,
        embedding_exclude_overlap=pretrained_pipeline.embedding_exclude_overlap,
        clustering=pretrained_pipeline.klustering, 
    )

    pipeline.instantiate({
        "clustering": {
            "threshold": pretrained_pipeline._pipelines['clustering']._instantiated['threshold'],
            "min_cluster_size": pretrained_pipeline._pipelines['clustering']._instantiated['min_cluster_size'],
            "method": pretrained_pipeline._pipelines['clustering']._instantiated['method'],
        },
        "segmentation": {
            "min_duration_off": pretrained_pipeline._pipelines['segmentation']._instantiated['min_duration_off'],
        },
    })

    
# Do hyperparameter tuning if requested
if hyper:
    # Load dev set
    dev_set = list(protocol.development())

    # Optimize hyperparameters
    from pyannote.pipeline import Optimizer
    
    pipeline.instantiate({
        "clustering": {
            "min_cluster_size": clussize,
        },
    })
    pipeline.to(torch.device("cuda"))
    optimizer = Optimizer(pipeline)
    iterations = optimizer.tune_iter(dev_set, show_progress=True)
    best_loss = 1.0
    for i, iteration in enumerate(iterations):
        print(f"Best clustering threshold so far: {iteration['params']['clustering']['threshold']}")
        if i > clusiter: break  # 50 iterations should give slightly better results

    # Change pipeline defult parameter (only thing that is optimized....)
    best_clustering_threshold = optimizer.best_params['clustering']['threshold']
    pipeline.instantiate({
        "clustering": {
            "threshold": best_clustering_threshold,
            "min_cluster_size": clussize,
        },
    })


# Input hyper parameters
if hyperset:
    pipeline.instantiate({
        "clustering": {
            "threshold": thresh,
            "min_cluster_size": clussize,
        },
    })


    
# Input groundtruths from rttm file
from pyannote.database.util import load_rttm
if groundtruths != 'test' :
    groundtruths = load_rttm(groundtruths)
if data == "rttm":
    hyps = load_rttm(args.rttmhyp)
    
# Define diarization metric
from pyannote.metrics.diarization import DiarizationErrorRate

if not args.nometric:
    metric = DiarizationErrorRate(collar=collar)

# Move pipeline to cuda
if args.rttmhyp == "":
    pipeline.to(torch.device("cuda"))

# Process test data
import os
for file in data_protocol :

    # Process file by file 

    print(file._store['uri'])
    
    if args.rttmhyp != "":
        if groundtruths != 'test':
            reference = groundtruths[file["uri"]]
        else:
            reference = file["annotation"]
        diarization = hyps[file["uri"]]
        metric(reference, diarization)
        continue

    if speakers == 0:
        diarization = pipeline(file)
    else:
        diarization = pipeline(file, min_speakers=speakers, max_speakers=speakers)

    if not args.nometric:
        if groundtruths != 'test':
            reference = groundtruths[file["uri"]]
        else:
            reference = file["annotation"]
        metric(reference, diarization)

#   Output rttm for each file if requested
    if args.rttm:
        if not os.path.exists(args.rttmdir) :
            try :
                os.makedirs(args.rttmdir)
            except :
                print ("Cannot create directory: " + args.rttmdir, file=sys.stderr)
                exit(1)
        basename = os.path.basename(file["uri"])        
        with open(args.rttmdir + "/" + basename.split(".")[0] + ".rttm", "w") as rttm:
            diarization.write_rttm(rttm)

# Compute and output metric
if not args.nometric:
    report = metric.report(display=True)

exit()
