# Store Visitor Data Solution Project
## CCTV를 활용한 매장 방문자 행동 데이터 솔루션

- Test video 구현 영상  (supermarket.mp4)

<a href='https://ifh.cc/v-ySbzKI' target='_blank'><img src='https://ifh.cc/g/ySbzKI.gif' border='0'></a>


```
python track.py --source supermarket.mp4 --yolo_weights yolov5/weights/crowdhuman_yolov5m.pt --classes 0  --show-vid --save-vid --save-txt
```



→ yolo v5 +deep sort 로 multiple object tracking

이때 yolo 에서는 COCO 데이터셋을 이용한 pretrained 된 weight 사용, 

우리가 하고자 task 에 맞춰  모든 class 를 분류하는 weight 말고  **사람만을 구별하는 weight** 사용 → 성능 향상

→ 처음 사람이이 감지가 되서 Tentative 상태로 돌입이고, 이후에 사람이라는 충분한 증거를 얻으면 사람이 입장한 것으로 간주  
ex.

```python
입장 id : 00  / 14:31:14.390320 
```
→ 첫 등장 bbox 를 기준으로 해당 id의 사람의 성별과 나이를 예측  
ex.

```python
id 00, 성별: male
```

```python
id 00, 나이: 20-39
```


→  처음 탐지가 되면 Tentative 상태로 돌입, 이후에 사람이라는 충분한 증거를 얻지 못하면 바로 삭제

ex. 

```python
00고객 삭제(Non-Person) : 00 / 14:31:51.538112
```

→ 탐지되고 있던 사람이 화면에서 일정 시간 (max age) 탐지되지 않으면 나간것으로 판단

```python
퇴장 id : 00 / 14:31:56.597224
```

→ 특정 구역 (화면에서 파란색 사각형 부분) 에 사람이 입장, 퇴장 할때

```python
00고객님 1섹션 입장 / 14:31:15.981412
```

```python
00고객님 1섹션 퇴장 / 14:31:23.672021
```




## DataFrame

모든 정보를 DataFrame 으로 자동 정리, 저장하여 .csv 파일로 자동 저장

- 총 체류시간 및 구역 별 체류시간 수집 가능



example  
![image](https://user-images.githubusercontent.com/84179578/142957803-69d52bbb-103c-4d64-9892-1a01a050d818.png)


----

## 구현 모델을  Real Data 에 적용

Data : 경기도 유명 쇼핑몰의 플리마켓 영상  




<a href='https://imgur.com/ciQepMo' target='_blank'><img src='https://i.imgur.com/252toyA.gif' border='0'></a>

![](https://i.imgur.com/252toyA.gif)


<a href="https://imgbb.com/"><img src="https://i.ibb.co/mDc2SkX/ezgif-com-gif-maker-4.gif" alt="ezgif-com-gif-maker-4" border="0"></a><br /><a target='_blank' href='https://poetandpoem.com/Edward-Dyson/Unredeemed'>unredeemed lyrics</a><br />

추출한 방문객 데이터 dataframe

![image](https://user-images.githubusercontent.com/84179578/145829108-5236c6ff-02de-4c90-8637-8fcae12d8c42.png)






----
## Reference

Multiple Object Tracking: A Literature Review
    

Real Time Pear Fruit Detection and Counting Using YOLOv4 Models and Deep SORT


__GitHub__  
[Yolov5_DeepSort__Pytorch - mikel-brostrom](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)

[Yolov5 - ultralytics](https://github.com/ultralytics/yolov5)