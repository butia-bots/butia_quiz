# butia_quiz package

## 1. Description

This package provides the ability for the robot DoRIS to answer previously known questions.

___
## 2. Requirements

You will need:
- [ROS Noetic](http://wiki.ros.org/noetic/Installation)
- pip

___
## 3. Download dependencies

To install the dependencies you will need to run the _install.sh_ with super user pemission

```
chmod +x install.sh
sudo ./install.sh
```

The dependencies will be installed.

___
## 4. Nodes

We develop a node to receive and find the question in a questions file.

The **butia_quiz_node** is a ROS Service that receives the question in text form. From there, this question is searched in the questions file and returns the corresponding answer. This search is performed using two-word comparison methods, such as the Levenshtein method. After finding the answer, it is sent using the same ROS Service.

___
## 5. Services and messages

We created a service to make possible the communication between ours packages.

### 5.1 Services

- ButiaQuizComm.srv receives a question as string and returns a answer as string.

___
## 6. Usage

The first use of the node may take a while as the models are being downloaded.

To start the node you must run:
```
roslaunch butia_quiz.launch
```

This launch will to load the config file and run the butia_quiz_node.

## License & Citation
If you find this package useful, consider citing it using:
```
@misc{butia_quiz,
    title={Butia Quiz Package},
    author={{ButiaBots}},
    howpublished={\url{https://github.com/butia-bots/butia_quiz/}},
    year={2022}
}

<p align="center"> 
  <i>If you liked this repository, please don't forget to starred it!</i>
  <img src="https://img.shields.io/github/stars/butia-bots/butia_quiz?style=social"/>
</p>
