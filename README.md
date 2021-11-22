# Store Visitor Data Solution Project
## CCTV를 활용한 매장 방문자 행동 데이터 솔루션

- 구현 영상

<a href='https://ifh.cc/v-ySbzKI' target='_blank'><img src='https://ifh.cc/g/ySbzKI.gif' border='0'></a>


```
python track.py --source supermarket.mp4 --yolo_weights yolov5/weights/crowdhuman_yolov5m.pt --classes 0  --show-vid --save-vid --save-txt
```



→ yolo v5 +deep sort 로 multiple object tracking

이때 yolo 에서는 COCO 데이터셋을 이용한 pretrained 된 weight 사용, 

우리가 하고자 task 에 맞춰  모든 class 를 분류하는 weight 말고  **사람만을 구별하는 weight** 사용 → 성능 향상

→ 처음 사람이 탐지 됐을때

```python
안녕하세요! 9고객님 / 14:31:14.390320 / 9
```

→  처음 탐지가 되면 Tentative 상태로 돌입, 이후에 사람이라는 충분한 증거를 얻지 못하면 바로 삭제

ex. 

```python
00고객 삭제(Non-Person) : 00 / 14:31:51.538112
```

→ 탐지되고 있던 사람이 화면에서 일정 시간 (max age) 탐지되지 않으면 나간것으로 판단

```python
00고객님, 안녕히 가세요! / 14:31:56.597224
```

→ 특정 구역 (화면에서 파란색 사각형 부분) 에 사람이 입장, 퇴장 할때

```python
2고객님 1섹션 입장 / 14:31:15.981412
```

```python
2고객님 1섹션 퇴장 / 14:31:23.672021
```




## DataFrame

모든 정보를 DataFrame 으로 자동 정리, 저장



example


----
## Reference

Multiple Object Tracking: A Literature Review
    

Real Time Pear Fruit Detection and Counting Using YOLOv4 Models and Deep SORT


[Yolov5_DeepSort__Pytorch](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)