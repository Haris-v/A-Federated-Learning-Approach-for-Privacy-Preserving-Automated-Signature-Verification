from pathlib import Path
from PIL import Image
import os
import random 


class Image_classification():

    ROOT_WINDOWS_FOLDER_PATH = Path(r"..\data\signatures")
    DOCKER_ROOT_WINDOWS_FOLDER_PATH = Path("code\data\signatures")

    DATA_CLASSIFICATION_FOLDER_PATH = Path(r"..\data\classification\signatures")
    DOCKER_DATA_CLASSIFICATION_FOLDER_PATH = Path("code\data\classification\signatures")

    DATA_CLASSIFICATION_FOLDER_PATH_RF = Path(r"..\data\classification\signaturesRF")
    DOCKER_DATA_CLASSIFICATION_FOLDER_PATH_RF = Path("code\data\classification\signaturesRF")

    # Path for docker
    # ROOT_DOCKER_FOLDER_PATH = Path("code/data/signatures")

    PNG_FILE_PREFIX = ".png"
    CLIENT_FOLDER_PREFIX = "client_"
    CLIENT_PARSE_DELIMETER = "_"

    def is_image(self, item):
        if item.endswith(self.PNG_FILE_PREFIX):
            return True
        else:
            return False
        


    def save_raw_images_by_client(self, files, from_root, to_root_folder):
        for item in files:
            print("Processing item: ", item)
            if self.is_image(item):
                client_number = item.split(self.CLIENT_PARSE_DELIMETER)[1]
                client_folder = self.CLIENT_FOLDER_PREFIX + client_number
                # list_parsed_root_path = from_root.split("\\")
                image_save_folder_path = to_root_folder.joinpath(client_folder)

                if not os.path.exists(str(image_save_folder_path)):
                    os.makedirs(image_save_folder_path)
                
                if not os.path.exists(image_save_folder_path / Path(item).name):
                    img = Image.open(Path(from_root) / item)
                    img.save(image_save_folder_path / Path(img.filename).name)
                    print("Image saved succesfully: ", img.filename)
                    img.close()
                else:
                    print("The file is already exists!")
                
                
                    
    def sort_images_by_clientid(self):
        for root, dirs, files in os.walk(self.ROOT_WINDOWS_FOLDER_PATH):
            print("Reading files from: ", root)
            self.save_raw_images_by_client(files, root, self.DATA_CLASSIFICATION_FOLDER_PATH)


    def sort_images_by_clientid_with_random_forgeries(self):
        client_counter = 1
        random_forgeries = []
        for root, dirs, files in os.walk(self.DATA_CLASSIFICATION_FOLDER_PATH):
            
            #Check if the image is random forgery or not
            if client_counter <= 31:
                self.save_raw_images_by_client_with_random_forgeries(files, root, self.DATA_CLASSIFICATION_FOLDER_PATH_RF, is_random_forgeries=False)
            else:
                for file in files:
                    root_and_files_path = Path(root).joinpath(Path(file))
                    random_forgeries.append(root_and_files_path)

                random.shuffle(random_forgeries)
            client_counter += 1
        
        root=None
        print(f"The total number of random forgery files: {len(random_forgeries)}")
        self.save_raw_images_by_client_with_random_forgeries(random_forgeries, root, self.DATA_CLASSIFICATION_FOLDER_PATH_RF, is_random_forgeries=True)
        

            
    def save_raw_images_by_client_with_random_forgeries(self, files, from_root, to_root_folder, is_random_forgeries):

        if not is_random_forgeries:
            for item in files:
                if self.is_image(item):
                    item_name_path = Path(item).name
                    image_save_folder_path = self.get_save_path(item, self.CLIENT_PARSE_DELIMETER, self.CLIENT_FOLDER_PREFIX, to_root_folder)

                    if not os.path.exists(image_save_folder_path):
                        os.makedirs(image_save_folder_path)
                    
                    self.save_the_image(from_root, item_name_path, image_save_folder_path, item_name_path)
        else:
            devided_files = self.devide_random_forgery_files_per_client(files)
            for i, client_files in enumerate(devided_files):
                number_of_client = i+1
                client_name = self.get_client_name(number_of_client)

                for index, item in enumerate(client_files):
                    number_of_rf_image = 24 + index + 1
                    
                    new_item_name = self.get_item_name(str(number_of_client), str(number_of_rf_image))
                    image_save_folder_path = self.DATA_CLASSIFICATION_FOLDER_PATH_RF.joinpath(Path(client_name))
                    print(f"Save the image: {item} , with the name of {new_item_name}, to the folder: {image_save_folder_path}")
                    self.save_the_image(from_root, item, image_save_folder_path, new_item_name)




    def get_save_path(self, item, delimiter, client_folder_prefix, to_root_folder):
        client_number = item.split(delimiter)[1]
        client_folder = client_folder_prefix + client_number
        # list_parsed_root_path = from_root.split("\\")
        return Path(to_root_folder).joinpath(Path(client_folder))
    

    def save_the_image(self, from_folder_path, item_name, to_folder_path, item_saved_name):
        
        if not os.path.exists(to_folder_path / item_name):

            if from_folder_path is not None:
                img = Image.open(Path(from_folder_path) / item_name) # or Image.open(from_folder_path + "\\" + item)
            else:
                img = Image.open(item_name) # or Image.open(from_folder_path + "\\" + item)
            
            img.save(to_folder_path / item_saved_name)
            print("Image saved succesfully: ", item_saved_name)
            img.close
        else:
            print("The file is already exists!")



    def devide_random_forgery_files_per_client(self, random_forgery_files):
        print(f"The total number of random forgery files: {len(random_forgery_files)}")
        chunk_size = round(len(random_forgery_files) // 30)  
        seperation_random_forgeries_per_client = [random_forgery_files[i:i + chunk_size] for i in range(0, len(random_forgery_files), chunk_size)]
        return seperation_random_forgeries_per_client
    

    def get_item_name(self,  number_of_client, number_of_rf_image):
        return "random_forgery_" + number_of_client + "_" + number_of_rf_image + self.PNG_FILE_PREFIX

    def get_client_name(self, number_of_client):
        if number_of_client >= 10:
            return self.CLIENT_FOLDER_PREFIX + str(number_of_client)
        else:
            return self.CLIENT_FOLDER_PREFIX + "0" + str(number_of_client)

classification_data = Image_classification()
classification_data.sort_images_by_clientid_with_random_forgeries()