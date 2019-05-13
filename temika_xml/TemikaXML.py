
class TemikaXML:
    """
    Class representing the Temika XML

    To use simply inherit from this class and use the functions provided

    Override output to change where the xml is saved.

    TODO:
        camera settings
        switching filters
        switching objectives
        more temika options
        better documentation
    """

    def __init__(self):
        self.opening()
        self.open_tag='temika'

    def output(self, txt):
        print(txt)

    def comment(self, txt):
        self.output(f"<!-- {txt} -->")

    def opening(self):
        self.output("<temika>")

    def closing(self):
        self.output("</temika>")

    def move_xy(self, distance, period, axis='x', mode='relative'):
        opd=False        
        if not self.open_tag == "microscope":
            opd=True
            self.output("<microscope>")
        self.output('<xystage axis="{}">'.format(axis))
        self.output('<enable>ON</enable>')
        self.output(f'<move_{mode}>{distance:.3f} {period}</move_{mode}>')
        self.output('</xystage>')
        if opd:
            self.output("</microscope>")

    def move_z(self, distance, period, mode='relative'):
        if mode not in ['relative', 'absolute']:
            print(f"INVALID mode {mode}")
            exit(-1)
        self.output('<microscope><eclipsetie>')
        self.output(f'<zdrive_move_{mode}>{distance:.3f} {period}'+f'</zdrive_move_{mode}>')
        self.output("</eclipsetie></microscope>")


    def move_relative(self, vec, period):
        """
        Take a 2D or 3D np array, move xy stage and z axis by the values
        """
        #TODO can we send xy move in 1 command
        if vec[0] != 0:
            self.move_xy(vec[0], period, 'x')
        if vec[1] != 0:
            self.move_xy(vec[1], period, 'y')
        if vec.shape[0] == 3 and vec[2] != 0:
            self.move_z(vec[2],period, 'relative')

    def set_led(self, channel, led, brightness):
        """
        Set channel:led to given brightness, switch to this led, disable other channel
        """
        self.output('<multiled device="microscope">')
        self.output('<trigger>EXTERNAL0</trigger>') #why?
        other_c= 1 if channel==0 else 0
        self.output('<enable channel="{}">OFF</enable>'.format(other_c))
        self.output('<set channel="{}">{:.3f}</set>'.format(channel, brightness))
        self.output('<number channel="{}">{}</number>'.format(channel, led))
        self.output(f'<enable channel="{channel}">ON</enable>')
        self.output('</multiled>')


    def _sec_to_timestring(self, secs):
        hours = int(secs // 3600)
        secs %= 3600
        mins = int(secs // 60)
        secs = secs%60
        return f"{hours}:{mins}:{secs:.3f}"

    def sleep(self, time, from_timestamp=-1):
        """
        wait time seconds
        if from_timestamp>=0 will wait from a timestamp
        """
        timestring=self._sec_to_timestring(time)
        if from_timestamp>=0:
            self.output(f'<sleep timestamp="{from_timestamp}">{timestring}</sleep>')
        else:
            self.output(f'<sleep>{timestring}</sleep>')

    def set_name(self, basename, append="DATE"):
        self.output("<save>")
        self.output(f"<basename>{basename}</basename>")
        if append:
            self.output(f"<append>{append}</append>")
        self.output("</save>")

    def open_camera_tag(self, camera="IIDC Point Grey Research Grasshopper3 GS3-U3-23S6M"):
        self.open_tag='camera'
        self.output(f'<camera name="{camera}">')

    #TODO: keep a proper stack, this assumes only 1 level depth
    def close_tag(self):
        tg=self.open_tag
        self.output(f'</{tg}>')
        self.open_tag='temika'

    def record_start(self, camera="IIDC Point Grey Research Grasshopper3 GS3-U3-23S6M"):
        self._output_camera(camera,'<record>ON</record>');

    def record_end(self, camera="IIDC Point Grey Research Grasshopper3 GS3-U3-23S6M"):
        self._output_camera(camera,'<record>OFF</record>');


    def light_path(self, light_path="L100"):
        """
        Set light path (EYE|L100|R100|L80)
        """
        self.output("<microscope>\n<eclipsetie>")
        self.output(f'<light_path>{light_path}</light_path>')
        self.output("</microscope>\n</eclipsetie>")

    def pfs_on(self):
        self.output("<microscope>\n<eclipsetie>")
        self.output('<pfs_enable>ON</pfs_enable>')
        self.output("</microscope>\n</eclipsetie>")

    def pfs_off(self):
        self.output("<microscope>\n<eclipsetie>")
        self.output('<pfs_enable>OFF</pfs_enable>')
        self.output("</microscope>\n</eclipsetie>")

    def pfs_offset(self, offset):
        self.output("<microscope>\n<eclipsetie>")
        self.output('<pfs_offset>{offset:.3f}</pfs_offset>')
        self.output("</microscope>\n</eclipsetie>")

    def wait_for_move(self, axis):
        """
        Wait for the stage to finish moving
        """
        self.output(f'<microscope><xystage axis="{axis}">')
        self.output('<wait_moving_end></wait_moving_end>')
        self.output('</xystage></microscope>')

    def _output_camera(self, camera, output):
        """
        Helper for outputing camera elements
        """
        if not self.open_tag=="camera":
            self.open_camera_tag(camera)
            self.output(output)
            self.close_tag()
        else:
            self.output(output)

    #TODO this is in  the wrong place
    def set_camera_trigger(self, trig='SOFTWARE', camera="IIDC Point Grey Research Grasshopper3 GS3-U3-23S6M"):
        self._output_camera(camera,'')
        #self._output_camera(camera, f'<trigger>SOFTWARE</trigger>')

    def trigger_camera(self, camera="IIDC Point Grey Research Grasshopper3 GS3-U3-23S6M"):
        self._output_camera(camera, '<send_trigger></send_trigger>')
