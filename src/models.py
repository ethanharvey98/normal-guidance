import torch
# Importing our custom module(s)
import layers

class InstanceLevelClassifier(torch.nn.Module):
    def __init__(self, in_features, out_features, kernel_size):
        super().__init__()

        self.instance_conv = layers.InstanceConv1d(in_features=in_features, kernel_size=kernel_size)
        self.clf = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=True)

    def forward(self, x, lengths):
        out = self.instance_conv(x, lengths)
        out = attn_logits = self.clf(out)
        attn_weights = torch.cat([
            torch.nn.functional.softmax(attn_logits_i, dim=0) 
            for attn_logits_i in torch.split(attn_logits, lengths)
        ])
        return out, attn_weights

class ClfPool(torch.nn.Module):
    def __init__(self, in_features, out_features, encoder=None, pooling="Max", neighbors=1):
        super().__init__()

        self.encoder = encoder
        self.clf = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=True)

        assert pooling in ["Max", "Mean", "ABMIL", "SmAP"]
        if pooling == "Max":
            self.pool = layers.Max()
        elif pooling == "Mean":
            self.pool = layers.Mean()
        elif pooling == "ABMIL":
            self.pool = layers.ABMIL(in_features=out_features)
        elif pooling == "SmAP":
            self.pool = layers.SmAP(in_features=out_features, neighbors=neighbors)

    def forward(self, x, lengths):
        if self.encoder is not None:
            x = self.encoder(x, lengths)
        out = self.clf(x)                                
        out, attn_weights = self.pool(out, lengths)
        return out, attn_weights
    
class PoolClf(torch.nn.Module):
    def __init__(self, in_features, out_features, encoder=None, pooling="Max", num_heads=8, neighbors=1):
        super().__init__()
        
        self.encoder = encoder

        assert pooling in ["Max", "Mean", "ABMIL", "TransMIL", "SmAP", "SmTAP"]
        if pooling == "Max":
            self.pool = layers.Max()
        elif pooling == "Mean":
            self.pool = layers.Mean()
        elif pooling == "ABMIL":
            self.pool = layers.ABMIL(in_features=in_features)
        elif pooling == "TransMIL":
            self.pool = layers.TransMIL(in_features=in_features, num_heads=num_heads)
        elif pooling == "SmAP":
            self.pool = layers.SmAP(in_features=in_features, neighbors=neighbors)
        elif pooling == "SmTAP":
            self.pool = layers.SmTAP(in_features=in_features, neighbors=neighbors)
            
        self.clf = torch.nn.Linear(in_features=in_features, out_features=out_features, bias=True)

    def forward(self, x, lengths):
        if self.encoder is not None:
            x = self.encoder(x, lengths)
        out, attn_weights = self.pool(x, lengths)
        out = self.clf(out)
        return out, attn_weights
    
class OnTheDesign(torch.nn.Module):
    def __init__(self, num_channels, num_classes, expansion=4, norm_type="Instance"):
        super().__init__()
        
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.expansion = expansion
        assert norm_type in ["Instance", "Batch"]
        self.norm_type = norm_type
        
        self.conv = torch.nn.Sequential()

        self.conv.add_module("conv0_s1", torch.nn.Conv3d(in_channels=self.num_channels, out_channels=4*self.expansion, kernel_size=1))

        if self.norm_type == "Instance":
            self.conv.add_module("lrn0_s1", torch.nn.InstanceNorm3d(num_features=4*self.expansion))
        elif self.norm_type == "Batch":
            self.conv.add_module("lrn0_s1", torch.nn.BatchNorm3d(num_features=4*self.expansion))
            
        self.conv.add_module("relu0_s1", torch.nn.ReLU(inplace=True))
        self.conv.add_module("pool0_s1", torch.nn.MaxPool3d(kernel_size=3, stride=2))

        self.conv.add_module("conv1_s1", torch.nn.Conv3d(in_channels=4*self.expansion, out_channels=32*self.expansion, kernel_size=3, padding=0, dilation=2))
        
        if self.norm_type == "Instance":
            self.conv.add_module("lrn1_s1", torch.nn.InstanceNorm3d(num_features=32*self.expansion))
        elif self.norm_type == "Batch":
            self.conv.add_module("lrn1_s1", torch.nn.BatchNorm3d(num_features=32*self.expansion))
            
        self.conv.add_module("relu1_s1", torch.nn.ReLU(inplace=True))
        self.conv.add_module("pool1_s1", torch.nn.MaxPool3d(kernel_size=3, stride=2))

        self.conv.add_module("conv2_s1", torch.nn.Conv3d(in_channels=32*self.expansion, out_channels=64*self.expansion, kernel_size=5, padding=2, dilation=2))
        
        if self.norm_type == "Instance":
            self.conv.add_module("lrn2_s1", torch.nn.InstanceNorm3d(num_features=64*self.expansion))
        elif self.norm_type == "Batch":
            self.conv.add_module("lrn2_s1", torch.nn.BatchNorm3d(num_features=64*self.expansion))
            
        self.conv.add_module("relu2_s1", torch.nn.ReLU(inplace=True))
        self.conv.add_module("pool2_s1", torch.nn.MaxPool3d(kernel_size=3, stride=2))

        self.conv.add_module("conv3_s1", torch.nn.Conv3d(in_channels=64*self.expansion, out_channels=64*self.expansion, kernel_size=3, padding=1, dilation=2))
        
        if self.norm_type == "Instance":
            self.conv.add_module("lrn3_s1", torch.nn.InstanceNorm3d(num_features=64*self.expansion))
        elif self.norm_type == "Batch":
            self.conv.add_module("lrn3_s1", torch.nn.BatchNorm3d(num_features=64*self.expansion))
            
        self.conv.add_module("relu3_s1", torch.nn.ReLU(inplace=True))
        self.conv.add_module("pool3_s1", torch.nn.MaxPool3d(kernel_size=5, stride=2))

        self.head = torch.nn.Sequential()
        self.head.add_module("fc0_s1", torch.nn.Linear(in_features=288*64*self.expansion, out_features=1024))
        self.head.add_module("relu4_s1", torch.nn.ReLU(inplace=True))
        self.head.add_module("fc1_s1", torch.nn.Linear(in_features=1024, out_features=self.num_classes))
        
    def forward(self, x):
        x = self.conv(x)
        x = self.head(x.flatten(start_dim=1, end_dim=-1))
        return x
    