from demo_class import VSS

class VSS_Evaluator:
    def __init__(self, vss_instance, ground_truth):
        """
        Initialize the evaluator with the VSS instance and ground truth data.

        :param vss_instance: An instance of the VSS class.
        :param ground_truth: A dictionary where keys are text queries and values are sets of relevant video names.
        """
        self.vss = vss_instance
        self.ground_truth = ground_truth

    def evaluate(self):
        """
        Evaluate the semantic_search function.

        :return: score from 0 to 1 representing the precentage of search it's correct
        """

        true_positives = 0
        
        for text_query, relevant_videos in self.ground_truth.items():
            retrieved_videos, _ = self.vss.semantic_search(text_query)
            top_retrieved_video = retrieved_videos[0]

            if top_retrieved_video in relevant_videos:
                true_positives += 1
            else:
                print(f"Fail Test:\nTest Query: {text_query}\nRelevant Videos: {relevant_videos}\nRetrieved Videos: {retrieved_videos}")
        score = true_positives / len(self.ground_truth)

        return score

testing_videos_path = "data/testing/"
vss = VSS(video_path= testing_videos_path)

# Define the ground truth data
ground_truth = {
    "forest": {"2021.07.04-Trees_Spinning.mov"},
    "new york city": {"911-memorial-1.mov"},
    "fun party": {"xmas_party_2023.mov", "2021.10.22-ghetto-ping-pong-pt-2.mov"},
    "airplane": {"210706_Plane_Window.mov"},
    "celebration": {"210729_Boston-Rally-Tunnel-2.mov", "xmas_party_2023.mov"},
    "crowd catching subway": {"Subway-2.mov"},
    "snow fighting!": {"2014-12-13_Staff_Snow_Trip_Mt_Baldy-50.mov"},
    "group of people having fun in an Arcade": {"dnbJamesKungFu.mov"},
    "playing pingpong on two tables": {"2021.10.22-ghetto-ping-pong-pt-2.mov"},
    "packing into a truck and moving": {"2021.06.10-POD_Packing_Timelapse.mov"}
}

evaluator = VSS_Evaluator(vss, ground_truth)
#os.remove("video_semantic_search/frame_embeddingss.safetensors")

# Evaluate the semantic search
evaluation_results = evaluator.evaluate()
print("Score:", evaluation_results)