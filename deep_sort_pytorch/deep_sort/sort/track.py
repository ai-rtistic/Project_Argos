# vim: expandtab:ts=4:sw=4

from datetime import datetime
import pandas as pd

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

    def update(self, kf, detection, class_id):
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
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
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
            print(f'{self.track_id}고객님, 안녕히 가세요! / {datetime.now().time()}')
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
    # image_size = 384x640
    # (650, 0), (900, 200)
    
    def in_placeA(self, coordinate):
        placeA = (650, 0, 900, 200) # (x1, y1, x2, y2)
        x = coordinate[0]
        y = coordinate[1]
        
        if (x > placeA[0] and x < placeA[2]):
            if (y > placeA[1] and y < placeA[3]):
                return True
        else:
            False
               
        
    def update_location(self, bboxes):
        before_coordinate = self.coordinate
        current_coordinate = (int((bboxes[2] + bboxes[0])/2), bboxes[3])
        
        if self.in_placeA(current_coordinate):
            self.current_locate = 1
        else:
            self.current_locate = 0
            
        if self.current_locate != self.before_locate:

            # dataframe
            df = pd.read_csv('dataframe/data.csv')

            if self.current_locate == 0:
                print(f'{self.track_id}고객님 {self.before_locate}섹션 퇴장 / {datetime.now().time()}')
                df.loc[df["person_id"]==self.track_id,"section_A_out"] = datetime.now().time()
                
            elif self.current_locate == 1:
                print(f'{self.track_id}고객님 {self.current_locate}섹션 입장 / {datetime.now().time()}')
                df.loc[df["person_id"]==self.track_id,"section_A_in"] = datetime.now().time()
            
            df.to_csv('dataframe/data.csv', index=False)


    
        self.before_locate = self.current_locate