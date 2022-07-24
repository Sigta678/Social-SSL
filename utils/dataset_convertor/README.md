# Dataset Preprocessing



## ◆ Dataset Format

This section will describe the two datasets, ETH/UCY and SDD, that were run in the experiment and the preprocessing we did on them. The data format accepted by our model is .npz, which is divided into five data items according to the key value: obsvs, preds, batches, times, idx_and_dist.

'batches' is used to indicate the start index and end index of the data contained in each frame;
'obsvs' and 'preds' save the coordinates of all agents in the increasing order of timestamps. Each 'obsvs' data contains the coordinates of 8 time points observed from the past to the present, and 'preds' contains the coordinates of the next 12 time points;
'times' has the same number of items as 'obsvs' and 'preds', and stores the timestamp of the corresponding frame;
'idx_and_dist' has the same number of items as 'batches', and for each agent per frame, stores the distance between this agent and other agents.




## ◆ Preprocessing Process

The following will describe how to convert the dataset into the .npz format accepted by our model. For both the two datasets, the preprocessing process is roughly as follows:
1. Collect all timestamps and corresponding coordinate positions that appear of each agent id.
2-1. Iterates over all timestamps covered by the dataset according to the set sampling frequency. For example, if the sampling frequency is set to 3, the starting position of the following trajectory sequence will skip its starting position to the third timestamp after the current trajectory sequence.
2-2. According to the data collected in step 1, for each agent, confirm whether there are past and future trajectories of a complete time period based on the current timestamp. If so, add the data to the dataset. For the judgment of the complete time period, we will look at the past 7 time points and look at the future 12 time points from the current timestamp (a total of 20 time points). The 'interval' hyperparameter control the gap of each two recorded timestamps will skip how many actual frames. For example, if the interval is set to 2, the next frame is 2 frames behind the current frame.
3. Check for the trajectory sequence data with the same starting timestamp, save the start index and end index of the frame in the data set.
4. For each agent in each frame, compute the Euclidean distance between that agent and other agents in the same frame.
5. Transform the coordinate data to the [0, 1] interval by normalization (Min-Max Scaling).




## ◆ Raw Dataset and Notes

Some details to note for each of the two datasets are as follows


ETH and UCY:

ETH dataset contains two scenes, ETH and HOTEL. UCY is another dataset which contains the ZARA1, ZARA2, and UNIV scenes. In many trajectory prediction studies, a total of 5 scenes in ETH and UCY are used to evaluate the leave-one-out performance of the model in the context of dense crowds. We use the txt file organized by Social-GAN as the raw dataset, which can be downloaded by this [site](https://www.dropbox.com/s/8n02xqv3l9q18r1/datasets.zip?dl=0&file_subpath=%2Fdatasets). It can be noted that it have been used in many papers, Social-STGCNN, STGAT, etc. We convert it to .npz format through the above process, using the same sampling frequency = 1 and interval = 1 as previous works.


Stanford Drone Dataset (SDD):

The SDD dataset is a large-scale dataset contains pedestrians, bicyclists, and vehicles using train-test split to evaluate the model performance. It can be downloaded directly on its [official website](https://cvgl.stanford.edu/projects/uav_data/). The original data format is similar to ETH/UCY and is stored in .txt files. Also, we convert it to .npz format through the previous process, using the same sampling frequency = 1 and interval = 12 as previous works.

