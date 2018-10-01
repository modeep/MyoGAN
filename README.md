# About MyoGAN

* The Deep Learning Project with MoDeep, Taeyeong Kim (InSpace Inc.)

* Generating Sign Language with MYO Sensor (EMG Data)

* Most using: **CGAN**, **MYO**, Data Preprocessing (RMS, etc.)

## Special Thanks for

* Sangmin Park (jigeria), the Idea provider.
  * Contact: jigeria@naver.com / jigeria114@gmail.com

* All MoDeep Team Members, leading the project.

## MyoGAN 진행 계획

### 1주차 (10. 01 ~ 10. 07)

1. **EMG Feature** 이해 -> 학습해보기 (학습이 잘 되면 계속 사용)

2. **EMG Feature**를 조정하는 방법 이해
    * DL이 아닌 것들: **RMS, Threshold**
    * DL: **LSTM, AutoEncoder**

3. 2번 항목을 이용해서 **Classifier** 만들어보기 (결과 공유할 것)

### 2주차 (10. 08 ~ 10. 14)

1. 1주차 3번의 **Classifier** 완성하고 Feature 조정법 결정

2. Feature 조정을 통해 **DCGAN** 성능 주시

### 3주차 (10. 15 ~ 10. 21)

1. **DCGAN**으로 만족할만한 결과 시도해볼 것

2. Condition을 넣는 작업을 시도해볼 것 -> **CGAN** 준비
    * **Hyperparam 튜닝** 노가다할 것

### 4주차 (10. 22 ~ 10. 28)

1. **CGAN** 완성할 것

### 5주차 이후 (10. 29 ~)

1. 완성한 **CGAN** 이용하여 **Hyperparam 튜닝**할 것

### 발표 준비

1. model이 무겁기 때문에 발표때 완벽한 시연은 사실상 힘듦
    * 대신 실시간으로 표시되는 **EMG를 시연할 것**