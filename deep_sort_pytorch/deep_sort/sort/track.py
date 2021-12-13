# vim: expandtab:ts=4:sw=4

from datetime import datetime
import pandas as pd

from gender_age_estimation.resnet_base import BottleNeck
from gender_age_estimation.resnet_base import ResNet
from gender_age_estimation.resnet_base import resnet50
from .prediction import predict_age_gender

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, mean, covariance, track_id, class_id, n_init, max_age, coordinate,
                 feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.class_id = class_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.bbox = [0, 0, 0, 0]
        
        self.coordinate = coordinate
        self.current_locate = 0
        self.before_locate = 0
        
        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get kf estimated current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The predicted kf bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def get_yolo_pred(self):
        """Get yolo prediction`.

        Returns
        -------
        ndarray
            The yolo bounding box.

        """
        return self.bbox.tlwh

    def increment_age(self):
        self.age += 1
        self.time_since_update += 1

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.increment_age()

    def update(self, kf, detection, class_id, f_idx=None, source=None): ### f_idx, source
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.bbox = detection
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)
        self.class_id = class_id

        self.hits += 1
        self.time_since_update = 0

        det = detection.to_tlbr()

        # resnet model
        if self.state == TrackState.Tentative and self.hits == self._n_init:
            
            print(f'입장 id : {self.track_id} / {datetime.now().time()}')
            print(f'left bot X: {det[0]},left bot Y:{det[3]}, right top X:{det[2]}, right top Y:{det[1]} ') 
            print(f'프레임 번호 : {f_idx}') ###
        

            ### resnet model  prediction
            pred = predict_age_gender(source, f_idx, det[0], det[3], det[2], det[1])
            gender = pred[0]
            print(f'성별: {gender}')

            age = pred[1]
            print(f'나이 : {age}')

            # dataframe
            new_data = [{"person_id":self.track_id,
                "entry_time": datetime.now().time(),
                "exit_time":None,
                "section_A_in":None,
                "section_A_out":None,
                "section_B_in":None,
                "section_B_out":None,
                "section_C_in":None,
                "section_C_out":None,
                "gender": gender,
                "age": age}]
            
            df = pd.read_csv('dataframe/data.csv')
            df = df.append(new_data,ignore_index=True)
            df.to_csv('dataframe/data.csv', index=False)

            self.state = TrackState.Confirmed

        elif self.state == TrackState.Tentative and self.hits > self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """

        # dataframe
        df = pd.read_csv('dataframe/data.csv')


        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
            print(f'{self.track_id}고객 삭제(Non-Person) : {self.track_id} / {datetime.now().time()}')
            df = df.drop(df[df["person_id"] == self.track_id].index)


        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted
            print(f'퇴장! id: {self.track_id} / {datetime.now().time()}')
            df.loc[df["person_id"]==self.track_id,"exit_time"] = datetime.now().time()

        
        df.to_csv('dataframe/data.csv', index=False)

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

    # bboxes = [x1, y1, x2, y2]

    
    def in_placeA(self, coordinate):
        placeA = (270, 150, 1000, 450) # (x1, y1, x2, y2)
        x = coordinate[0]
        y = coordinate[1]
        
        if (x > placeA[0] and x < placeA[2]):
            if (y > placeA[1] and y < placeA[3]):
                return True
        else:
            False

    def in_placeB(self, coordinate):
        placeB = (50, 480, 780, 840) # (x1, y1, x2, y2)
        x = coordinate[0]
        y = coordinate[1]
        
        if (x > placeB[0] and x < placeB[2]):
            if (y > placeB[1] and y < placeB[3]):
                return True
        else:
            False

    def in_placeC(self, coordinate):
        placeC = (900, 520, 1800, 900) # (x1, y1, x2, y2)
        x = coordinate[0]
        y = coordinate[1]
        
        if (x > placeC[0] and x < placeC[2]):
            if (y > placeC[1] and y < placeC[3]):
                return True
        else:
            False
               
        
    def update_location(self, bboxes):
        before_coordinate = self.coordinate
        current_coordinate = (int((bboxes[2] + bboxes[0])/2), bboxes[3])
        
        if self.in_placeA(current_coordinate):
            self.current_locate = 1
        elif self.in_placeB(current_coordinate):
            self.current_locate = 2
        elif self.in_placeC(current_coordinate):
            self.current_locate = 3
        else:
            self.current_locate = 0
        
        if self.current_locate != self.before_locate:
            
            # dataframe
            df = pd.read_csv('dataframe/data.csv')

            if self.before_locate == 0 and self.current_locate == 1:
                print(f'{self.track_id}고객님 {self.current_locate}섹션 입장 / {datetime.now().time()}')
                df.loc[df["person_id"]==self.track_id,"section_A_in"] = datetime.now().time()

            elif self.before_locate == 0 and self.current_locate == 2:
                print(f'{self.track_id}고객님 {self.current_locate}섹션 입장 / {datetime.now().time()}')
                df.loc[df["person_id"]==self.track_id,"section_B_in"] = datetime.now().time()

            elif self.before_locate == 0 and self.current_locate == 3:
                print(f'{self.track_id}고객님 {self.current_locate}섹션 입장 / {datetime.now().time()}')
                df.loc[df["person_id"]==self.track_id,"section_C_in"] = datetime.now().time()

            elif self.before_locate == 1 and self.current_locate == 0:
                print(f'{self.track_id}고객님 {self.before_locate}섹션 퇴장 / {datetime.now().time()}')
                df.loc[df["person_id"]==self.track_id,"section_A_out"] = datetime.now().time()

            elif self.before_locate == 2 and self.current_locate == 0:
                print(f'{self.track_id}고객님 {self.before_locate}섹션 퇴장 / {datetime.now().time()}')
                df.loc[df["person_id"]==self.track_id,"section_B_out"] = datetime.now().time()

            elif self.before_locate == 3 and self.current_locate == 0:
                print(f'{self.track_id}고객님 {self.before_locate}섹션 퇴장 / {datetime.now().time()}')
                df.loc[df["person_id"]==self.track_id,"section_C_out"] = datetime.now().time()
                

            
            df.to_csv('dataframe/data.csv', index=False)

            self.before_locate = self.current_locate