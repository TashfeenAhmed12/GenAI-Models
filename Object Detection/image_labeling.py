from PIL import Image, ImageDraw

def label_image(raw_image, detections, output_path=None, resize_dims=None):
    """
    Labels the image with bounding boxes and labels based on detection results.
    
    Parameters:
    - raw_image (PIL.Image.Image): The raw image object.
    - detections (list of dicts): List of detection results. Each dict should contain 'box' and 'label'.
    - output_path (str, optional): If provided, the labeled image will be saved to this path.
    - resize_dims (tuple, optional): A tuple (width, height) to resize the image. If None, no resizing is done.
    
    Returns:
    - Image object with labels drawn.
    """
    # Store the original image dimensions
    original_width, original_height = raw_image.size
    
    # Create a copy of the original image
    image_with_labels = raw_image.copy()
    
    # Resize the image if resize_dims is provided
    if resize_dims:
        image_with_labels = image_with_labels.resize(resize_dims)
    
    # Get the new image dimensions
    new_width, new_height = image_with_labels.size
    
    # Calculate scaling factors
    x_scale = new_width / original_width
    y_scale = new_height / original_height
    
    # Create a draw object
    draw = ImageDraw.Draw(image_with_labels)
    
    # Iterate over the detection results
    for detection in detections:
        # Get the bounding box coordinates
        box = detection['box']
        xmin, ymin, xmax, ymax = box['xmin'], box['ymin'], box['xmax'], box['ymax']
        
        # Scale the bounding box coordinates according to the resizing
        xmin = int(xmin * x_scale)
        ymin = int(ymin * y_scale)
        xmax = int(xmax * x_scale)
        ymax = int(ymax * y_scale)
        
        # Draw a rectangle around the object
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline='red', width=2)
        
        # Add the label text
        label = detection['label']
        draw.text((xmin, ymin), label, fill='red')
    
    # Save the labeled image if an output path is provided
    if output_path:
        image_with_labels.save(output_path)
    
    return image_with_labels
