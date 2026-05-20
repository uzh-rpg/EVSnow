import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

class Events:
    def __init__(self, x=None, y=None, t=None, p=None):
        #initialize events
        if x is None:
            self.x = []
        else:
            self.x = x
        if y is None:
            self.y = []
        else:
            self.y = y
        if t is None:
            self.t = []
        else:
            self.t = t
        if p is None:
            self.p = []
        else:
            self.p = p
        if x is not None:
            self.num_events = len(x)
        else:
            self.num_events = 0

    def __len__(self):
        return self.num_events
     
    def add_event(self, x, y, t, p):
        #add one event to list:
        self.x.append(x)
        self.y.append(y)
        self.t.append(t)
        self.p.append(p)
        self.num_events += 1
    
    def add_events(self, events):
        #add one event to list:
        self.x = np.append(self.x, events.x)
        self.y = np.append(self.y, events.y)
        self.t = np.append(self.t, events.t)
        self.p = np.append(self.p, events.p)
        self.num_events += events.num_events

    def read_events(self, event_file, start_event=0, num_events=None):
        #read events stored as npy file
        events = np.load(event_file)
        if num_events is not None:
            events = events[start_event:start_event+num_events]
        else:
            events = events
        self.x = events[:,0]
        self.y = events[:,1]
        self.t = events[:,2]
        self.p = events[:,3]
        self.num_events = len(self.x)
    
    def read_events_from_hdf5(self, event_file, start_event=0, num_events=None):
        import h5py
        f = h5py.File(str(event_file), 'r')
        events = f
        if num_events is not None:
            self.x = events['x'][start_event:start_event+num_events]
            self.y = events['y'][start_event:start_event+num_events]
            self.t = events['t'][start_event:start_event+num_events]
            self.p = events['p'][start_event:start_event+num_events]
            
        else:
            self.x = events['x'][start_event:]
            self.y = events['y'][start_event:]
            self.t = events['t'][start_event:]
            self.p = events['p'][start_event:]
        # sort events by timestamps
        sort_idx = np.argsort(self.t)
        self.x = self.x[sort_idx]
        self.y = self.y[sort_idx]
        self.t = self.t[sort_idx]
        self.p = self.p[sort_idx]

        self.num_events = len(self.x)

    def get_events_for_pixel(self, x,y):
        #get events for a pixel
        pixel_events = Events()
        for i in range(self.num_events):
            if self.x[i] == x and self.y[i] == y:
                pixel_events.add_event(self.x[i], self.y[i], self.t[i], self.p[i])
        return pixel_events

    def get_events_for_time(self, start_time, end_time):
        #get events for a time range
        mask = (self.t>=start_time) & (self.t<end_time)
        return Events(self.x[mask], self.y[mask], self.t[mask], self.p[mask])

    def get_n_events(self, start_id , n):
        #get n events starting from start_id
        n_events = Events()
        for i in range(start_id, start_id + n):
            n_events.add_event(self.x[i], self.y[i], self.t[i], self.p[i])
        return n_events


    def to_dict(self):
        return {"x": list(self.x), "y": list(self.y), "t": list(self.t), "p": list(self.p)}
    
    @staticmethod
    def render(events, shape, img:np.ndarray):
        if img is not None:
            bg_image = img.copy()
        else:
            bg_image = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)

        y = [int(i) for i in events.y]
        x = [int(i) for i in events.x]
        # invert x for real world data, because with beam splitter, the x axis is inverted compared to the image coordinates. 
        # For synthetic data, this is not needed.
        # x = [shape[1]-1 - int(i) for i in events.x]
        if len(x)>0:
            # p_color is red if p==-1 otherwise blue
            p_color = [[255,0,0] if i == 1 else [0,0,255] for i in events.p]
            bg_image[y, x, :] = p_color
        return bg_image

class VoxelGrid():
    def __init__(self, bins, w, h):
        self.bins = bins
        self.w = w
        self.h = h
    
    def _bil_w(self, x, x_int):
        return 1 - np.abs(x_int- x)
    
    def _draw_xy_to_voxel_grid(self, voxel_grid, x, y, b, value):
        if x.dtype == np.uint16:
            self._draw_xy_to_voxel_grid_int(voxel_grid, x, y, b, value)
            return

        x_int = x.astype("int32")
        y_int = y.astype("int32")
        for xlim in [x_int, x_int + 1]:
            for ylim in [y_int, y_int + 1]:
                weight = self._bil_w(x, xlim) * self._bil_w(y, ylim)
                self._draw_xy_to_voxel_grid_int(voxel_grid, xlim, ylim, b, weight * value)

    def _draw_xy_to_voxel_grid_int(self, voxel_grid, x, y, b, value):
        H, W, B = voxel_grid.shape
        mask = (x >= 0) & (y >= 0) & (x < W) & (y < H)
        np.add.at(voxel_grid, (y[mask], x[mask], b[mask]), value[mask])


    def save_event_image(self, events, filename):
        img = np.zeros((self.w, self.h, 3), dtype=np.uint8)
        img = Events.render(events, [self.w, self.h], img)
        cv2.imwrite(filename, img)

    def events_to_voxel_grid(self, events):
        """
        Build a voxel grid with trilinear interpolation in the time and x,y domain from a set of events.
        """
        events.p = events.p.astype(np.int8)
        events.p[events.p == 1] = 1  # Convert polarities to +1 / -1
        events.p[events.p == 0] = -1  # Convert polarities to +1 / -1
        assert events.p.min() == -1 and events.p.max() == 1, "Polarity values are not in the expected range of [-1, 1]"
        pos_grid = np.zeros((self.h, self.w, self.bins), np.float32)
        
        # normalize the event timestamps so that they lie between 0 and num_bins
        t0_us = events.t[0]
        t1_us = events.t[-1]
        deltaT = t1_us - t0_us
        B = self.bins
        if deltaT == 0:
            deltaT = 1.0
        x, y, t = events.x, events.y, events.t
        # x = self.w - x #-> only for real world
        t_norm = (B -1) * (t - t0_us) / deltaT
        t_norm_int = t_norm.astype("int32")
        for tlim in [t_norm_int, t_norm_int+1]:
            mask = (tlim >= 0) & (tlim < B)
            self._draw_xy_to_voxel_grid(pos_grid, x[mask], y[mask], tlim[mask], events.p[mask])    
        return pos_grid

    @staticmethod
    def normalize_voxel_grid(voxel_grid):
        mask = np.nonzero(voxel_grid)
        # mask = np.ones_like(voxel_grid, dtype=bool)
        if mask[0].size > 0:
            mean, stddev = voxel_grid[mask].mean(), voxel_grid[mask].std()
            if stddev > 0:
                voxel_grid[mask] = (voxel_grid[mask] - mean) / stddev
        return voxel_grid
    
def load_events_voxelgrid(event_file, voxel_file, res, bins=10, use_cache=True):
    if os.path.exists(voxel_file) and use_cache:
        try:
            voxel_grid = np.load(voxel_file)
        except:
            print("voxel not found, using empty", voxel_file)
            voxel_grid = np.zeros((res[0], res[1], bins))
    else:
        fullevents = Events()
        fullevents.read_events_from_hdf5(event_file)
        before_filter_num = fullevents.num_events
        #FOR DSEC DATA
        exxp_ts = 10000 + fullevents.t[0]
        # exxp_ts = fullevents.t[-1]
        events = Events()
        events.x = fullevents.x[fullevents.t<=exxp_ts]
        events.y = fullevents.y[fullevents.t<=exxp_ts]
        events.p = fullevents.p[fullevents.t<=exxp_ts]
        events.t = fullevents.t[fullevents.t<=exxp_ts]
        print("Removed events: ", before_filter_num - len(events.t))
        print("Events left: ", len(events.t))
        print("Events all duration: {} ms".format((events.t[-1]-events.t[0])/1000))
        vgrid = VoxelGrid(bins=bins, w=res[0], h=res[1])
        if len(events.t)>0 :
            print(np.sum(events.p))
            if np.sum(events.p)==0:
                voxel_grid = np.zeros((res[0], res[1], bins))
            else:
                voxel_grid = vgrid.events_to_voxel_grid(events)  
        else:
            voxel_grid = np.zeros((res[0], res[1], bins))
        os.makedirs(os.path.dirname(voxel_file), exist_ok=True)
        np.save(str(voxel_file), voxel_grid)

    voxel_grid = VoxelGrid.normalize_voxel_grid(voxel_grid)
    voxel_grid[-50:,:,:] = 0
    return voxel_grid

