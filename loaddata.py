import zipfile

extract_to_directory = './traditionalMLApp/data/traffic_sign_detection'
# Extract the zip file
with zipfile.ZipFile("./traditionalMLApp/data/traffic_sign_detection.zip", 'r') as zip_ref:
    zip_ref.extractall(extract_to_directory) 