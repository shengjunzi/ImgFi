import paddle
import paddle.nn as nn

class FeatureExtractor(nn.Layer):
    def __init__(self,numclass=16):
        super(FeatureExtractor, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2D(3, 64, 3, 2, 1),  
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.MaxPool2D(2), 
         
            nn.Conv2D(64, 128, 3, 2, 1),  
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.MaxPool2D(2),  

            nn.Conv2D(128, 256, 3, 2, 1),  
            nn.BatchNorm2D(256),
            nn.ReLU(),
            nn.MaxPool2D(2),  
    
            nn.Conv2D(256, 512, 3, 1, 1),  
            nn.BatchNorm2D(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2D(1),
           
        )
        self.layer = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, numclass)
        )
    def forward(self, x):
        x = self.conv(x).squeeze()
        x=self.layer(x)
        return x

        
