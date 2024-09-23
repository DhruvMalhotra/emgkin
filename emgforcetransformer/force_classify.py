import torch

def classify(values: torch.Tensor, num_classes: int, value_range: tuple):
    """
    Classifies input tensor into logits over num_classes buckets.
    Args:
        values (torch.Tensor): Input tensor of any shape.
        num_classes (int): Number of classes.
        value_range (tuple): Tuple (min_value, max_value) representing the range of possible values.
    Returns:
        torch.Tensor: Output tensor with an additional last dimension for num_classes logits.
    """
    min_value, max_value = value_range
    values_normalized = (values - min_value) / (max_value - min_value)
    class_labels = (values_normalized * (num_classes - 1)).long()  # Scale to the number of classes
    logits = torch.clamp(class_labels, min=0, max=num_classes - 1)
    return logits

def collapse(logits: torch.Tensor, num_classes: int, value_range: tuple):
    """
    Collapses logits back to a single value using the mean of the highest logit class.
    Args:
        logits (torch.Tensor): Logits tensor with last dimension as num_classes.
        num_classes (int): Number of classes.
        value_range (tuple): Tuple (min_value, max_value) representing the range of possible values.
    Returns:
        torch.Tensor: Collapsed tensor with original shape, without the last num_classes dimension.
    """
    min_value, max_value = value_range
    class_indices = torch.argmax(logits, dim=-1)  # Get the class with the highest logit
    class_means = (class_indices.float() + 0.5) / num_classes  # Map to the center of each bucket
    values = class_means * (max_value - min_value) + min_value
    return values
