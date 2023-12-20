import time


class CustomTimeCount():
    def __init__(self) -> None:
        self.begin_time=time.time()
        self.duration=0
    def getTimePassed(self):
        current_time=time.time()
        self.duration+=current_time-self.begin_time
        self.begin_time=current_time
        return self.duration
    def getTimePassedAsStr(self):
        duration=self.getTimePassed()
        sec=duration%60
        duration/=60
        mins=duration%60
        hour=duration/60
        return f"{int(hour)},{int(mins)},{int(sec)}"
class CustomLogger():
    def __init__(self,log_file:str,) -> None:
        self.time_counter=CustomTimeCount()
        self.log_path=log_file
    def log(self,log_header:str,log_str:str,to_stdout:bool=True):
        output=f"[{log_header}({self.time_counter.getTimePassedAsStr()})]: {log_str}"
        with open(self.log_path, 'a') as f:
            f.write(output)
            f.write("\n")
        if to_stdout:
            print(output)