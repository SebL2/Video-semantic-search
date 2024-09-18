import torch
import ast
import chromadb
from PIL import Image
import open_clip
import os   
import glob
import ffmpeg
from safetensors import safe_open
from kmeans_pytorch import kmeans
from safetensors.torch import save_file

class VSS:
    def __init__(self, 
                 video_path: str = os.path.dirname(os.path.abspath(__file__)) + "/../videos/",
                 frame_embeddings_st_path: str = os.path.dirname(os.path.abspath(__file__)) + "/frame_embeddingss.safetensors"
                 ):
        """
        Initialize the VSS class.

        :param video_path: Path to the directory containing video files. Defaults to a 
                        directory named 'videos' in the parent directory of the script.
        :param frame_embeddings_st_path: Path to the safetensor file where frame embeddings
                                        will be stored. Defaults to 'frame_embeddings.safetensors'
                                        in the same directory as the script.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.frames_path = os.path.dirname(os.path.abspath(__file__)) + "/videoData/"
        self.client = chromadb.PersistentClient(path=os.path.dirname(os.path.abspath(__file__)) + "/chromaDB/")
        self.collection = None  
        self.video_path = video_path
        self.videos_names_list = self._get_videos_names_list()
        # print('Frames path:', frames_path) 
        # print('Frame embeddings path:', frame_embeddings_st_path)
        self.frame_embeddings_st_path = frame_embeddings_st_path
        self.frame_embeddings = {}
        self.centroids = None
        self.centroidMap = self._generate_centroid_map()
        self._cluster_videos_with_centroids(num_clusters=2)
        
    def _cluster_videos_with_centroids(self, num_clusters=4):
        self.cluster_ids_x, self.cluster_centers = kmeans(
            X=self.centroids, num_clusters=num_clusters, distance='cosine', device=self.device
        )

    def _get_videos_names_list(self):
        return os.listdir(self.video_path)

    def _embed_text(self, text: str) -> torch.Tensor:
        """
        Convert text to its according embedding

        :param text: A word or sentence to create embedding with
        :return The according embedding for the input text
        """
        with torch.no_grad():
            text_encoded = open_clip.tokenize(text).to(self.device)
            text_embedding = self.model.encode_text(text_encoded)
        return text_embedding
    
    def _embed_frames(self, frames: list[tuple]) -> torch.Tensor:
        """
        Convert a list of frames(images) to a Tensor of embeddings

        :param frames: A list of tuple that contains image path
        and opened image (path, image)
        :return `Tensor` of shape (num of images, 512), containing 
        the embedding for all images in the list
        """
        frame_embeddings = []
        for frame in frames:
            with torch.no_grad():
                frame_preprocessed = (self.preprocess(frame[1])
                                         .unsqueeze(0)
                                         .to(self.device))
                
                frame_embedding = self.model.encode_image(frame_preprocessed)

            frame_embeddings.append(frame_embedding.squeeze())
        
        return torch.stack(frame_embeddings)


    
    def _get_frames(self, pathOnly: bool = False):
        """
        Get a list of tuples containing the image paths and opened image object
        or only a list of image paths

        :param pathOnly: When set to turn, only return a list of all image paths
        :return A `list` of (image path, opened image) or image path
        """
        filepaths = os.listdir(self.frames_path)
        frames = []
        for filepath in filepaths:
            if filepath.endswith(".png"):
                if pathOnly:
                    frames.append(filepath)
                else:
                    frames.append(
                        (filepath, 
                         Image.open(f"{self.frames_path}/{filepath}", mode='r'))
                    )
     
        return frames

    def _get_frame_embeddings(self) -> torch.Tensor:
        """
        Load the Tensor stored in safetensor file if the file exist, else generate 
        the Tensor and dump it into a safetensor file

        :return `Tensor` of all frame(image) embeddings
        """
        if os.path.exists(self.frame_embeddings_st_path):
            with safe_open(self.frame_embeddings_st_path,
                           framework="pt",
                           device=self.device) as f:
                print("Loading frame embeddings...")
                print(f.get_tensor("embeddings"))
                return f.get_tensor("embeddings")
        else:
            frames = self._get_frames()
            print('Creating frame embeddings...')
            frame_embeddings = self._embed_frames(frames)
            
            print('Saving frame embeddings to file directory...')
            frame_embeddings_to_save = {'embeddings' : frame_embeddings}
            save_file(frame_embeddings_to_save,
                      self.frame_embeddings_st_path)

            return frame_embeddings
        
    def _generate_embeddings_from_vid(self):
        '''
        Generates embeddings for each individual video and returns a list
        of embeddings
        '''
        if len(os.listdir(self.frames_path))>0:
            files = glob.glob(f"{self.frames_path}*")
            for file_name in files:
                os.remove(file_name)

        file_to_save = {}
        # filepath = self.videos_names_list
        self.collection = self.client.get_or_create_collection(name="test")
        for file_name in self.videos_names_list:
            if file_name.endswith(".mov"):
                (
                    ffmpeg
                    .input(self.video_path + file_name)
                    .output(f"{self.frames_path}/%d.png",vf="fps=1")
                    .run()
                ) 
            frames = self._get_frames()
            embeddings = self._embed_frames(frames)
            
            test = torch.Tensor.tolist(embeddings)
            self.collection.add(
                documents=[str(test)],
                ids = [file_name]
            )
            # file_to_save[file_name] = embeddings

            files = glob.glob(f"{self.frames_path}/*")
            for file_name in files:
                os.remove(file_name)
        # save_file(file_to_save, self.frame_embeddings_st_path)
        # return file_to_save
        
    def _generate_centroid_map(self):
        '''
        Generates a centroid from the list of embeddings for each video and
        create a dictionary/map that maps the centroid to the video embedding
        :return: A dictionary where keys are centroids (tensors) and values are 
        lists of frame embeddings (tensors) for each video.
        '''
        # find norm of each vector, average, use as centroid
        # or, find which vector has the closest norm to the average norm
        if not self.client.get_or_create_collection("test"):
            self.frame_embeddings = self._generate_embeddings_from_vid()
        else:
            self.collection = self.client.get_or_create_collection("test")
            for name in self.videos_names_list:
                key =torch.FloatTensor(ast.literal_eval(self.collection.get(ids=[name])["documents"][0]))
                self.frame_embeddings[name] = key
            # with safe_open(self.frame_embeddings_st_path,
            #                framework="pt",
            #                device=self.device) as f:
            #     for key in self.videos_names_list:
            #         self.frame_embeddings[key] = f.get_tensor(key)            
                  
        centroidMap = {}
        for name in self.frame_embeddings.keys():
            centroid_v = [0] * 512
            for vector in self.frame_embeddings[name]:
                # add all frame embeddings
                for component in range(len(vector)):
                    centroid_v[component] += vector[component]
                for i in range(len(centroid_v)):
                    centroid_v[i]/=512 
            if self.centroids == None:
                self.centroids = torch.as_tensor(centroid_v).unsqueeze(0)
            else:
                self.centroids = torch.cat((self.centroids, torch.as_tensor(centroid_v).unsqueeze(0)), dim=0)
            #print(self.centroids)
            centroidMap[torch.as_tensor(centroid_v)] = self.frame_embeddings[name]

        return centroidMap
    
    # def _compute_cloest_cluster_center(self, text_embedding):
    #     similarities = torch.cosine_similarity(text_embedding, self.cluster_centers)
    #     return torch.argmax(similarities)

    def chromaDB_generate(self):
        try:
            self.collection = self.client.get_collection("embeddings")
            self.client.delete_collection("embeddings")
            self.collection = self.client.create_collection("embeddings")
        except:
            self.collection = self.client.create_collection("embeddings")
        
        for file_name in self.videos_names_list:
            self.collection.add(
                documents = [file_name],
                ids = [file_name]
            )

    def chromaDB_search(self, text: str):
        results = self.collection.query(
            query_texts = [text],
            n_results = 1
        )
        return results["documents"][0]
        

    def semantic_search(self, text: str):
        """
        Generate the embedding for input text and calculate similarity between
        the input text and all kmeans centers, then calculate similarity between 
        centroids belong to that center. Return top five related image path and 
        their according similarities

        :param text: text that are used to get the most related image
        :return `tuple` of a `list` of top five most similar video names, and 
        Tensor containing their according similarities
        """
        print('Performing semantic search')
        # create text embeddings
        text_embedding = self._embed_text(text)
        # calculate index of the cloest center with the text_embedding
        cloest_center = self._compute_cloest_cluster_center(text_embedding)
        # filter out centroids that doesn't belong to that center
        mask = self.cluster_ids_x == cloest_center
        cloest_centroids = self.centroids[mask]
        # calculate similarities for centroids that belong to that center
        similarities = torch.cosine_similarity(text_embedding, cloest_centroids)
        # get top five indices for similarities and vidoe names
        top_five_videos_indices = similarities.argsort(descending=True)[:5]
        cloest_centroids_video_name_indices = torch.nonzero(mask, as_tuple=False).squeeze()
        top_five_video_names_indices = cloest_centroids_video_name_indices[top_five_videos_indices]
        return ([self.videos_names_list[i.item()] for i in top_five_video_names_indices],
                similarities[top_five_videos_indices])
    def get_shortest(self, input: str):
        '''
            Gets the shortest embedding/vector given a text query input from
            the centroid map of video embeddings
        '''
        inp_embedding = self._embed_text(input)
        lowest_distance = float("inf")
        lowest_vector = None
        for centroid in self.centroidMap.keys():
            
            new_min = 1-(torch.cosine_similarity(inp_embedding,centroid).item())
            if new_min < lowest_distance:
                lowest_distance = new_min
                lowest_vector = centroid
                
        #self.frame_embeddings key: video file name, values : embedding
        #centroidmap, key : singular vector, value : embedding
        for video_file in self.frame_embeddings.keys(): #key is video file name
            if self.frame_embeddings[video_file].equal(self.centroidMap[lowest_vector]):
                print(video_file) 
                # return (key,lowest_vector)

testing = VSS(video_path="video_semantic_search/testing/")
query_text = "woods"
print(testing.get_shortest("traffic"))

# import chromadb
# chroma_client = chromadb.PersistentClient()
# # switch `create_collection` to `get_or_xcreate_collection` to avoid creating a new collection every time
# collection = chroma_client.get_or_create_collection(name="my_collection")
# # switch `add` to `upsert` to avoid adding the same documents every time
# collection.upsert(
#     documents=[
#         "This is a document about pineapple",
#         "This is a document about oranges"
#     ],
#     ids=["id1", "id2"]
# )
# results = collection.query(
#     query_texts=["This is a query document about florida"], # Chroma will embed this for you
#     n_results=2 # how many results to return
# )
# print(results)