# 1. track.py 안에 있는 print들 다 # 풀어야함
# 2. dataset -> loadimage에 print 부분 # 풀어야함
# 3. tracker.py 안에 _initiate_track 메소드 print 지워야함
# 4. tracker.py 안에 Max Age 70 -> 40으로 낮췄음 다시 70으로 하던가 적당한 숫자로 바꾸셈
# 5. track.py 안에 mark_missed 메소드에 있는 print 추가했음




######### 알아낸 사실
# 처음 탐지가 되면 Tensative 상태로 돌입, 이후에 충분한 증거를 얻을 때까지 self.hit 증가. 근데 만약 금방 사라져버리면 바로 delete됨
# 특정 self.hit를 넘기지 못하면 그냥 바로 삭제(잘못 탐지된걸로 간주)

# 새로운 track id 생겨나면 tracker.py에 있는 _initiate_track 메소드를 실행함 / 그래서 print 확인해보면 이때 첫 시작
# track id가 제거되는 경우는 2가지 / 탐지 되자마자 증거 불충분으로 delete 또는 잘 가다가 안 보여서 age 많이 먹고 delete / track.py에 있는 mark_missed 확인해보면 됨.
#