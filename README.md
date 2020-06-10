## My contributions to the CS3244 project, Detecting Cyberbullying in Singapore context

I used the built in NaiveBayes model in scikit-learn, and fastAI's ULMFiT transfer learning model to train on the general dataset, 
which was from Kaggle, and a local dataset, which my team self-sourced from 
many different Singaporean websites, such as STOMP and HardwareZone. 

For the ULMFiT model, I implemented gradual unfreezing where all but the last layer was initially frozen, the unfrozen layers fine-tuned for one epoch, then the next layer unfrozen, and so on until all layers were trained.

I also implemented discriminative fine-tuning with specific learning rates for each layer - lower rates for layers near the input layers and higher rates for the later layers. This is so that there will not be many changes for the input layers as it has learned more general features, and the later layers will be able to learn the detailed features faster.
