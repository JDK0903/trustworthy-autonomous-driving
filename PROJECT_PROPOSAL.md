# Project proposal

Source: [supplied PDF](docs/PROJECT_PROPOSAL.pdf). Text extracted from the submitted proposal; the PDF is authoritative for formatting.

## Page 1

Project Proposal 
Interpretable and Risk-Aware Autonomous Driving Decisions 
Jaykumar Kabra (DAL169429), Shreyas Ranga (DAL277849) 
Concept Bottleneck Models with uncertainty aware abstention 
Problem and Motivation 
Autonomous systems often use neural networks that take sensor data or images and directly produce a decision. That 
can work well when the input looks similar to the training data, but it creates a problem when the model is wrong: it is 
hard to tell what part of the reasoning failed. A driving model might choose GO, for example, without giving us a clear 
way to tell whether it missed a pedestrian, misunderstood the road, or was simply uncertain. For a safety -related 
system, accuracy by itself is not enough. We also want the system to expose useful information about its decision and 
to avoid acting confidently when the input is unclear. 
Our project will build a small autonomous -driving decision prototype around this idea. Instead of predicting a driving 
action directly from an image, the model will first predict a few human-readable concepts, such as whether a pedestrian 
is present, whether a vehicle is present, whether the road is clear, and whether an obstacle appears close. A simple 
decision model will use those concepts to choose STOP, SLOW, or GO. We will also add an abstention option, 
meaning the system can output REQUEST HUMAN CONTROL when important concepts are too uncertain. The goal 
is not to build a complete self-driving system. The goal is to test whether interpretability and uncertainty -aware refusal 
can make a simplified autonomous decision pipeline easier to inspect and safer when it is unsure. 
Related Work 
The main idea comes from Concept Bottleneck Models (CBMs ). A CBM predicts meaningful intermediate concepts 
before predicting the final label, which makes it possible to inspect those concepts and even correct them at test time. 
Their experiments showed that changing incorrect concept predictions can improve the final prediction. Our project 
uses the same basic idea, but applies it to a simplified road -scene decision problem where the concepts are things a 
person can understand directly, such as pedestrian presence or road clearance. 
The second part of our project is related to selective prediction. Geifman and El -Yaniv (2019) studied models with a 
reject option, where the model is allowed to avoid making a prediction on examples it does not trust. We use a much 
simpler version of that idea: if a safety -related concept is too uncertain, the system requests human control rather than 
forcing STOP, SLOW, or GO. This also connects to the calibration work of Guo et al. (2017), which showed that 
modern neural networks can be overconfident and  that predicted probabilities should not automatically be treated as 
reliable confidence values. Because of this, we will choose our abstention threshold using validation data rather than 
assuming that a raw probability is perfectly calibrated. 
We also want to see what happens when the input quality gets worse. Hendrycks and Dietterich (2019) showed that 
common image corruptions such as blur, noise, and weather effects can cause large drops in neural -network 
performance. We will use a few simple image corruptions for the same reason: a trustworthy system should behave 
sensibly when the image is less clear than normal. Finally, Rudin (2019) argues that high -stakes systems should use 
interpretable models when possible instead of relying only on post-hoc explanations of black boxes. That motivates our 
use of explicit concepts and a shallow decision tree rather than making Grad -CAM or another heatmap the main 
explanation. 
Methodology 
We plan to use a small subset of the Cityscapes road -scene dataset, starting with roughly 500 to 1,000 images. 
Cityscapes includes semantic annotations, so we can derive concept labels without manually labeling every image. The 
first three concepts will be pedestrian_present, vehicle_present, and cyclist_present. We will also create road_clear and 
obstacle_close using simple rules based on the lower-center part of the segmentation mask. For example, a large object 
occupying the lower-center region can be treated as a proxy for a close obstacle. This is not true depth estimation, and 
we will state that limitation clearly. 
A pretrained ResNet18 will be used as the concept predictor. Its final layer will be replaced with five outputs, one for 
each concept, and the model will be trained as a multi -label classifier. We will first freeze most of the pretrained 
network and train the new output layer. If needed, we can fine -tune the last ResNet block. The predicted concepts will 

## Page 2

then be given to a shallow decision tree with a small maximum depth so that its decision path remains readable. The 
final driving actions will be STOP, SLOW, and GO. 
On top of the CBM, we will add a simple risk rule. If a safety -critical concept falls inside an uncertainty range, the 
model will abstain and return REQUEST HUMAN CONTROL. The exact threshold will be selected using the 
validation set. We will also train a second ResNet18 that predicts STOP, SLOW, or GO directly from the image. This 
will serve as the end-to-end baseline. The important comparison is not only which model has the highest accuracy, but 
what happens when the models make mistakes and when the input becomes difficult. 
A second experiment will test concept intervention. When the CBM makes an incorrect final decision because one of 
its concept predictions is wrong, we will replace that concept with its ground -truth value and run the decision tree 
again. For example, if pedestrian_present is predicted as false but the annotation says true, we can correct that one 
concept and check whether the action changes from GO or SLOW to STOP. This gives us a direct way to measure 
whether the interpretability built into the model is ac tually useful rather than only producing an explanation after the 
fact. 
Evaluation Plan 
We will evaluate the project on a held -out test set and keep threshold selection and other tuning on the validation set. 
For the concept model, we will report precision, recall, and F1 score for each of the five concepts. For the final decision 
task, we will report accuracy and macro F1 for the standard CBM and the end-to-end baseline. 
The main trustworthy-AI metrics will focus on failure behavior. Intervention Success Rate will measure the percentage 
of incorrect actions that become correct after an incorrect concept is replaced with its ground -truth value. For the risk-
aware model, Coverage will measure the percentage of test examples on which the model actually makes a driving 
decision instead of abstaining. We will also report the error rate on the non-abstained examples and count severe errors 
where the ground -truth action is STOP bu t the model chooses GO without abstaining. This lets us see the trade -off 
between making fewer decisions and making safer decisions. 
Finally, we will create corrupted versions of part of the test set using a small number of common changes such as blur, 
darkness, and image noise. We will compare how much the action accuracy drops and whether the risk -aware model 
abstains more often on those harder inputs. A useful result would be that the risk -aware model gives up some coverage 
but reduces the number of severe confident errors. 
Course Themes 
This project combines at least two major trustworthy -AI themes. The first is interpretability and explainability: the 
Concept Bottleneck Model exposes intermediate concepts and the shallow decision tree gives a readable path from 
those concepts to the action. The second is uncertainty, robustness, and safe failure: the abstention mechanism allows 
the system to hand control back when the visual evidence is unclear, and the corruption tests examine how behavior 
changes under degraded inputs. Human oversight i s also part of the project because concept intervention allows a 
person to correct an intermediate prediction and observe the effect on the final decision. 
The project is intentionally small enough for a semester. We are not implementing a simulator, real steering or throttle 
control, LiDAR, reinforcement learning, or a new object detector. If the full plan becomes too large, the minimum 
version will keep the CBM, the end-to-end baseline, one abstention rule, and the concept-intervention experiment. 
References 
• Geifman, Y., & El-Yaniv, R. (2019). SelectiveNet: A Deep Neural Network with an Integrated Reject Option. 
Proceedings of the 36th International Conference on Machine Learning, 97, 2151-2159. 
• Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. 
Proceedings of the 34th International Conference on Machine Learning, 70, 1321-1330. 
• Hendrycks, D., & Dietterich, T. (2019). Benchmarking Neural Network Robustness to Common Corruptions 
and Perturbations. International Conference on Learning Representations. 
• Koh, P. W., Nguyen, T., Tang, Y. S., Mussmann, S., Pierson, E., Kim, B., & Liang, P. (2020). Concept 
Bottleneck Models. Proceedings of the 37th International Conference on Machine Learning, 119, 5338-5348. 
• Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use 
interpretable models instead. Nature Machine Intelligence, 1, 206-215. 

