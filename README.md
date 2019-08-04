# mnist_dnn
This repo provides a deep neural network with my own mnist dataset

본 Repository는 Tensorflow를 기반으로 mnist image dataset을 불러와 MLP(Multi-layered perceptron)를 이용하여 Classification을 구현한 코드이다.

보통 이론적인 내용을 배운후에, mnist dataset 예제를 단 2줄의 코드로 다운받아 학습이 되는 과정을 실습한다.

이제 어느정도 딥러닝 알고리즘을 본의의 데이터에 사용할 준비가 되었다고 생각할 때, 
가장 당황스러운 것은 본인의 데이터를 어떻게 불러오고, 학습이 가능하도록 정리하는지에 관한 것이다. 

본 repository는 mnist dataset을 본인의 데이터라고 가정하고, 데이터를 불러오는 것부터 학습이 가능하도록 이미지 데이터를 정리하고, 학습하고, accuracy를 확인하는 것들을 포함한다.
mnist dataset이 아닌 다른 이미지 데이터를 사용하려면 소스코드상에서 이미지의 크기(28x28, 784)만 바꾸면 될 것이다.

# Update:
2019/07/27



코드를 실행하기 위해서는 다음의 명령어를 입력한다.

python3 dnn_mnist.py

코드에 대한 자세한 설명은 다음의 블로그를 참조하시길 바란다.

https://intuitive-robotics.tistory.com/53


