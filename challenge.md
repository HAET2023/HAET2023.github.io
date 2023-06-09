# Competition

In parallel with the workshop, we organize a competition. It is not required to participate in the competition to have a paper accepted at the workshop.

# Main Guidelines

The aim of the competition is to achieve the best possible accuracy in a classification task given a limited training time. The best three teams will receive awards and will be invited to present their solution during the workshop.

In more details, the training will be performed on an Nvidia V100 GPU with 32GB memory, running with an Intel(R) Xeon(R) Gold 6230 CPU @ 2.10GHz processor, with 12GB of RAM memory, with an allowed training time of 10 minutes.

The competition will be preceded by an open trial phase where participants can stress their methods using a publicly available dataset (CIFAR10). But the final dataset used for ranking the submissions will remain secret.

To participate, the candidates must send a pdf describing their method with their code as supplementary material (more details in the next section). The description can be short (1 page is enough).

# Details and Important Dates

During the trial phase, participants can download all needed code from [here](https://github.com/eghouti/HAET-2021-competition-baseline-code). For this phase, contestants can use the provided code with CIFAR10 dataset and should try to achieve the best possible accuracy using limited training time. 

Once the participants have their submission ready, they can submit it using the [CMT](https://cmt3.research.microsoft.com/HAET2021) link, as choose Competition as a category. Before the workshop, submissions will be evaluated using a different dataset (unknown to the participants). The submissions will be evaluated by running the code for a duration of 10 minutes. The evaluation consists in classifying 10 classes with inputs made of 32 by 32 RGB images. 500 inputs per class are available during training and 100 per class fortest. Note that during the evaluation, we considere data pre-processing, neutwork architecture and the way the network is trained (optimizer, learning rate... etc) the participant will provide, and we only change the input training and test dataset.

As the purpose of the competition is to evaluate the ability to quickly train systems, it is forbidden to rely on pretrained models or additional data. Ranked submissions will be checked and discarded if they do so.

Note that during evaluation, we will kill the running process after 10 minutes of training. So participants are advised to save their model regularly to avoid missing the deadline.

Methods will be ranked according to the accuracy they achieve after 10 minutes of training.

# Dates

- Competition submission deadline: 1 March 2021 (11:59 pm UTC-11).
- Workshop: 7 May 2021.
