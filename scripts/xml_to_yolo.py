# xml_to_yolo.py
import os
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import shutil

# --- CONFIGURATION ---

# 1. Input Directories (Where raw data is)
ANNOTATION_DIR = 'archive/annotations/'
IMAGE_DIR = 'archive/images/'

# 2. Output Directories (The final structure YOLO needs)
YOLO_ROOT = 'yolo_data/'
LABELS_DIR = os.path.join(YOLO_ROOT, 'labels')
IMAGES_DIR = os.path.join(YOLO_ROOT, 'images')

# Define the split ratio (e.g., 80% train, 20% val)
VAL_RATIO = 0.20 

# 3. Class Mapping (MUST match your label_map.pbtxt/YOLO needs)
# YOLO requires IDs starting from 0.
CLASS_MAPPING = {
    'with_mask': 0,
    'without_mask': 1,
    'mask_weared_incorrect': 2
}

def convert_box(size, box):
    """Converts PASCAL VOC bounding box to normalized YOLO format."""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    
    # Calculate center coordinates, width, and height
    center_x = (box[0] + box[2]) / 2.0  # (xmin + xmax) / 2
    center_y = (box[1] + box[3]) / 2.0  # (ymin + ymax) / 2
    width = box[2] - box[0]
    height = box[3] - box[1]

    # Normalize
    center_x = center_x * dw
    center_y = center_y * dh
    width = width * dw
    height = height * dh

    # Return as string formatted to 6 decimal places
    return (center_x, center_y, width, height)


def process_annotations():
    """Reads XMLs, converts boxes, and saves to YOLO TXT format."""
    
    # --- Create necessary folders ---
    for d in [os.path.join(LABELS_DIR, 'train'), os.path.join(LABELS_DIR, 'val'), 
              os.path.join(IMAGES_DIR, 'train'), os.path.join(IMAGES_DIR, 'val')]:
        os.makedirs(d, exist_ok=True)

    # 1. Get list of all XML filenames
    all_xml_files = [f for f in os.listdir(ANNOTATION_DIR) if f.endswith('.xml')]
    
    # 2. Split the list of filenames (images/annotations must be split together)
    train_xml_files, val_xml_files = train_test_split(
        all_xml_files, test_size=VAL_RATIO, random_state=42
    )

    print(f"Total files: {len(all_xml_files)}")
    print(f"Training set size: {len(train_xml_files)}")
    print(f"Validation set size: {len(val_xml_files)}")
    
    # --- Conversion Function ---
    def convert_and_copy(xml_files, split_type):
        for xml_filename in xml_files:
            xml_path = os.path.join(ANNOTATION_DIR, xml_filename)
            
            # --- Convert Annotation (XML -> TXT) ---
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Get image dimensions (for normalization)
            size = root.find('size')
            img_width = int(size.find('width').text)
            img_height = int(size.find('height').text)
            
            yolo_txt_content = []
            
            for member in root.findall('object'):
                class_name = member.find('name').text
                
                # Check for label consistency
                if class_name not in CLASS_MAPPING:
                    print(f"Warning: Skipping unknown class '{class_name}' in {xml_filename}")
                    continue
                    
                class_id = CLASS_MAPPING[class_name]
                
                # Extract pixel coordinates
                bndbox = member.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                
                # Convert to YOLO format
                yolo_box = convert_box((img_width, img_height), (xmin, ymin, xmax, ymax))
                
                # Create the YOLO line: [id center_x center_y width height]
                yolo_line = f"{class_id} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}\n"
                yolo_txt_content.append(yolo_line)

            # Save the YOLO annotation file (.txt)
            txt_filename = xml_filename.replace('.xml', '.txt')
            txt_path = os.path.join(LABELS_DIR, split_type, txt_filename)
            with open(txt_path, 'w') as f:
                f.writelines(yolo_txt_content)

            # --- Copy Image ---
            img_filename = xml_filename.replace('.xml', '.png')
            
            # Handle possible .jpg files (try/except for robustness)
            img_src_path = os.path.join(IMAGE_DIR, img_filename)
            if not os.path.exists(img_src_path):
                img_filename = xml_filename.replace('.xml', '.jpg')
                img_src_path = os.path.join(IMAGE_DIR, img_filename)
            
            img_dest_path = os.path.join(IMAGES_DIR, split_type, img_filename)
            shutil.copy(img_src_path, img_dest_path)


    # 3. Run the conversion for both splits
    convert_and_copy(train_xml_files, 'train')
    convert_and_copy(val_xml_files, 'val')
    
    print("\n YOLO data conversion complete!")
    print("Files saved to the 'yolo_data/' directory.")


if __name__ == '__main__':
    # Ensure pandas and scikit-learn are installed in this environment!
    import pandas # just to make sure
    from sklearn.model_selection import train_test_split
    process_annotations()