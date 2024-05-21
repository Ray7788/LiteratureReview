## Elements of building generalist agents: 
1) an environment that supports a multitude of tasks and goals,
2) a large-scale database of multimodal knowledge, and
3)  a flexible and scalable agent architecture.

##  Contributions
 simulation suite, knowledge bases, algorithm implementation, and pretrained model

### MINEDOJO provides
1) convenient APIs on top of Minecraft that standardize task specification, world settings, and **agent’s observation/action spaces**.
2) a benchmark suite that consists of thousands of natural language-prompted tasks
3) a novel agent evaluation protocol using a large video-language model pre-trained on Minecraft YouTube videos
4) MINEDOJO features a massive collection of 730K+ YouTube videos with time-aligned transcripts,6K+ free-form Wiki pages, and 340K+ Reddit posts with multimedia contents (Fig. 3)
5) train a video-text contrastive model in the spirit of CLIP [92], which associates natural language
subtitles with their time-aligned video segments.
