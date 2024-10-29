import toupcam, io, tifffile, rawdev, motor
import numpy as np

CAM_NOT_FOUND = 1
CAM_INIT_FAILED = 2
CAM_OPEN_FAILED = 3

class Cam:
    def __init__(self):
        self.hcam = None
        self.binned = None
        self.rawbuf = None
        self.lowres_buf = None
        self.total = 0
        self.save_capture = False
        self.has_new = False
        self.curr_shutter_speed = 350
        self.curr_contrast = None
        self.start_focus = False
        self.curr_direction = 'CW'
        self.TEC_target = -50
        self.expo_time = 10000
        self.delay_frames = 2
        self.LCG_threshold = 160
        self.HCG_threshold = 120
        self.conversion_gain = 0

    # the vast majority of callbacks come from toupcam.dll/so/dylib internal threads
    # full-res callback with binning for image capture
    @staticmethod
    def cameraCallback(nEvent, ctx):
        if nEvent == toupcam.TOUPCAM_EVENT_IMAGE:
            ctx.CameraCallback(nEvent)

    # low-res fast-readout callback for preview/exposure/focusing
    @staticmethod
    def previewCallback(nEvent, ctx):
        if nEvent == toupcam.TOUPCAM_EVENT_IMAGE:
            ctx.PreviewCallback(nEvent)

    def CameraCallback(self, nEvent):
        if nEvent == toupcam.TOUPCAM_EVENT_IMAGE:
            try:
                self.hcam.PullImageV3(self.rawbuf, 0, 48, 0, None)
                self.total += 1
                rawdata = np.frombuffer(self.rawbuf, dtype='uint16')
                self.binned = rawdev.process_raw_binning(rawdata)

                if self.delay_frames == 0:
                    min_pixel = np.min(rawdata)
                    max_pixel = np.max(rawdata)
                    print("MIN: "+str(min_pixel)+" MAX: "+str(max_pixel))

                    if(min_pixel < self.LCG_threshold):
                        self.expo_time += 100
                        print("curr expo time = "+str(self.expo_time))
                        try:
                            self.hcam.put_ExpoTime(self.expo_time)
                        except toupcam.HRESULTException as ex:
                            print('set expo failed, hr=0x{:x}'.format(ex.hr & 0xffffffff))
                    elif(max_pixel > 65500):
                        self.expo_time -= 100
                        try:
                            self.hcam.put_ExpoTime(self.expo_time)
                        except toupcam.HRESULTException as ex:
                            print('set expo failed, hr=0x{:x}'.format(ex.hr & 0xffffffff))
                    self.delay_frames = 2

                self.has_new = True
            except toupcam.HRESULTException as ex:
                print('pull full-res image failed, hr=0x{:x}'.format(ex.hr & 0xffffffff))

    def PreviewCallback(self, nEvent):
        if nEvent == toupcam.TOUPCAM_EVENT_IMAGE:
            try:
                self.hcam.PullImageV3(self.rawbuf, 0, 24, 0, None)
                self.total += 1
                rawdata = np.frombuffer(self.rawbuf, dtype='uint8')
                self.binned = rawdev.process_low_res(rawdata)

                self.has_new = True
            except toupcam.HRESULTException as ex:
                print('pull low-res image failed, hr=0x{:x}'.format(ex.hr & 0xffffffff))
    #test funnction to check that cam module is starting correctly
    def run(self):
        a = toupcam.Toupcam.EnumV2()
        if len(a) > 0:
            print('{}: flag = {:#x}, preview = {}, still = {}'.format(a[0].displayname, 
                                a[0].model.flag, a[0].model.preview, a[0].model.still))
            for r in a[0].model.res:
                print('\t = [{} x {}]'.format(r.width, r.height))
            self.hcam = toupcam.Toupcam.Open(a[0].id)
            print('high-fullwell ={:x}'.format(a[0].model.flag & toupcam.TOUPCAM_FLAG_HIGH_FULLWELL))
            print('raw bit size ={:x}'.format(a[0].model.flag & toupcam.TOUPCAM_FLAG_RAW16))
            if self.hcam:
                try:
                    toupcam.Toupcam.put_Option(self.hcam, toupcam.TOUPCAM_OPTION_FAN, self.hcam.FanMaxSpeed())
                    toupcam.Toupcam.put_Option(self.hcam, toupcam.TOUPCAM_OPTION_TEC, 1)
                    toupcam.Toupcam.put_Option(self.hcam, toupcam.TOUPCAM_OPTION_TECTARGET, self.TEC_target)

                except toupcam.HRESULTException as e:
                    print('init failed, err=0x{:x}'.format(e.hr & 0xffffffff))
                    self.hcam.Stop()
                    self.hcam.Close()
                    self.hcam = None
                    self.buf = None

                print("####...REACHED! CLOSING...#####")
                #self.hcam.Stop()
                #self.hcam.Close()
                #self.hcam = None
                #self.buf = None
            else:
                print('failed to open camera')
        else:
            print('no camera found')

    # pull full resolution for image capture
    def pull_high_res(self):
        #if self.rawbuf:
            try:
                self.hcam.Stop()
                self.hcam.put_eSize(0)

                width, height = self.hcam.get_Size()
                rawbufsize = (width * 2) * height
                print('image size: {} x {}, bufsize = {}'.format(width, height, rawbufsize))
                self.rawbuf = bytes((int)(rawbufsize))

                self.hcam.put_Option(toupcam.TOUPCAM_OPTION_RAW, 1)
                #self.hcam.put_Option(toupcam.TOUPCAM_OPTION_BINNING, 0x82)
                self.hcam.put_Option(toupcam.TOUPCAM_OPTION_PIXEL_FORMAT, toupcam.TOUPCAM_PIXELFORMAT_RAW16)
                self.hcam.put_Option(toupcam.TOUPCAM_OPTION_CG, self.conversion_gain)
                self.hcam.put_Option(toupcam.TOUPCAM_OPTION_LOW_NOISE, 1)
                #disable auto exposure
                self.hcam.put_AutoExpoEnable(0)
                self.hcam.put_ExpoTime(self.expo_time)
                print("raw option = ", self.hcam.get_Option(toupcam.TOUPCAM_OPTION_PIXEL_FORMAT))
                print("CONVERSION GAIN option = ", self.hcam.get_Option(toupcam.TOUPCAM_OPTION_CG))
                self.hcam.StartPullModeWithCallback(self.cameraCallback, self)

            except toupcam.HRESULTException as ex:
                print('failed to start hi-res mode, hr=0x{:x}'.format(ex.hr & 0xffffffff))

    # pull low-res fast-readout for preview
    def pull_low_res(self):
        #if self.rawbuf:
            try:
                self.hcam.Stop()
                self.hcam.put_eSize(1)

                width, height = self.hcam.get_Size()
                rawbufsize = width * height
                print('image size: {} x {}, bufsize = {}'.format(width, height, rawbufsize))
                self.rawbuf = bytes(rawbufsize)

                self.hcam.put_Option(toupcam.TOUPCAM_OPTION_RAW, 1)
                self.hcam.put_Option(toupcam.TOUPCAM_OPTION_PIXEL_FORMAT, toupcam.TOUPCAM_PIXELFORMAT_RAW8)
                self.hcam.put_Option(toupcam.TOUPCAM_OPTION_CG, self.conversion_gain)
                #disable auto exposure
                self.hcam.put_AutoExpoEnable(0)
                self.hcam.put_ExpoTime(self.expo_time)
                print("raw option = ", self.hcam.get_Option(toupcam.TOUPCAM_OPTION_PIXEL_FORMAT))
                print("CONVERSION GAIN option = ", self.hcam.get_Option(toupcam.TOUPCAM_OPTION_CG))
                self.hcam.StartPullModeWithCallback(self.previewCallback, self)

            except toupcam.HRESULTException as ex:
                print('failed to start lo-res mode, hr=0x{:x}'.format(ex.hr & 0xffffffff))

    def get_temp(self):
        return self.hcam.get_Temperature()/10

    def set_temp(self, target):
        self.hcam.put_Temperature(target*10)

    def get_shutter_speed(self):
        return self.hcam.get_ExpoTime()
    
    def set_shutter_speed(self, time):
        self.hcam.put_ExpoTime(time)

    def capture(self):
        self.save_capture = True
    
    def shutdown(self):
        self.hcam.Stop()
        self.hcam.Close()
        self.hcam = None
        self.buf = None
        print("##### CAM EXITED NORMALLY #####")

if __name__ == '__main__':
    app = Cam()
    app.run()