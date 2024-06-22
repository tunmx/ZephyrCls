import torch
import torch.nn as nn
import torchvision.models as models


class ResNet34Classifier(nn.Module):
    def __init__(self, class_num, use_weights=True):
        super(ResNet34Classifier, self).__init__()
        # Load the pretrained ResNet34 model
        self.resnet34 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if use_weights else None)

        # Replace the final fully connected layer
        num_ftrs = self.resnet34.fc.in_features
        self.resnet34.fc = nn.Linear(num_ftrs, class_num)

    def forward(self, x):
        return self.resnet34(x)


# Example usage
if __name__ == "__main__":
    # Define the number of classes
    class_num = 10

    # Create an instance of the classifier
    model = ResNet34Classifier(class_num=class_num, use_weights=True)

    # Print the model architecture
    print(model)

    # Create a dummy input tensor
    inputs = torch.randn(1, 3, 224, 224)

    # Forward pass
    outputs = model(inputs)

    # Print the output shape
    print(outputs.shape)
