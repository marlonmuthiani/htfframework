import threading
import os
from enum import Enum
import time
import pandas as pd
from backtest.input_configuration import TrainInputConfiguration
from configuration import operative_system

DEFAULT_JVM_WIN = '-Xmx8000M'
DEFAULT_JVM_UNIX = '-Xmx8000M'


class TrainLauncherState(Enum):
    created = 0
    running = 1
    finished = 2


class TrainLauncher(threading.Thread):
    VERBOSE_OUTPUT = False
    if operative_system == 'windows':
        DEFAULT_JVM = DEFAULT_JVM_WIN
    else:
        DEFAULT_JVM = DEFAULT_JVM_UNIX

    def __init__(
        self,
        train_input_configuration: TrainInputConfiguration,
        id: str,
        jar_path='Backtest.jar',
        jvm_options: str = DEFAULT_JVM,
    ):
        threading.Thread.__init__(self)
        self.train_input_configuration = train_input_configuration
        self.id = id
        self.jar_path = jar_path
        self.jvm_options = jvm_options
        self.jvm_options += f' -Dlog.appName=train_model'  # change log name
        self.state = TrainLauncherState.created
        # https://github.com/eclipse/deeplearning4j/issues/2981
        self.task = 'java %s -jar "%s"' % (self.jvm_options, self.jar_path)

    def run(self):
        self.state = TrainLauncherState.running
        # Generate a full valid backtest JSON structure for the Java process
        import json
        file_content = json.dumps({
            "backtest": {
                "startDate": "20250831 06:00:00",
                "endDate": "20250831 20:00:00",
                "instrument": "ethbtc_binance",
                "delayOrderMs": 0,
                "feesCommissionsIncluded": True,
                "seed": 0,
                "multithreadConfiguration": "single_thread",
                "outputPath": "output/",
                "dataPath": "D:\\git repos\\HFTFramework\\data",
                "logLevel": "INFO"
            },
            "algorithm": {
                "algorithmName": "AlphaAvellanedaStoikov",
                "parameters": {
                    "dummyAgent": 0,
                    "quantity": 0.001,
                    "seed": 28220,
                    "riskAversion": 0.1,
                    "kDefault": 0.5,
                    "aDefault": 0.5,
                    "sigmaDefault": -1,
                    "spreadCalculation": "Avellaneda",
                    "kCalculation": "Pct",
                    "positionMultiplier": 335.35,
                    "spreadMultiplier": 2.0,
                    "ui": 1,
                    "minOrderSize": 0.001,
                    "maxOrderSize": 1.0,
                    "maxPosition": 10.0,
                    "minSpread": 0.01,
                    "maxSpread": 10.0,
                    "enableInventory": True,
                    "enableHedge": False,
                    "scoreEnum": "total_pnl",
                    "horizonMinMsTick": 1000,
                    "horizonTicksPrivateState": 1,
                    "midpricePeriodWindowAction": [5],
                    "riskAversionAction": [0.1],
                    "skewAction": [0.0, 1.0, -1.0],
                    "riskAversionAction": [0.06327, 0.31635, 0.6327, 0.94905],
                    "midpricePeriodWindowAction": [24, 50],
                    "changeKPeriodSecondsAction": [60.0],
                    "midpricePeriodSeconds": 26,
                    "minPrivateState": -1,
                    "maxPrivateState": -1,
                    "numberDecimalsPrivateState": 3,
                    "horizonTicksPrivateState": 5,
                    "minMarketState": -1,
                    "maxMarketState": -1,
                    "numberDecimalsMarketState": 7,
                    "horizonTicksMarketState": 10,
                    "minCandleState": -1,
                    "maxCandleState": -1,
                    "numberDecimalsCandleState": 3,
                    "horizonCandlesState": 4,
                    "horizonMinMsTick": 0,
                    "stateColumnsFilter": [],
                    "scoreEnum": "asymmetric_dampened_pnl",
                    "stepSeconds": 5,
                    "rlPort": 2122,
                    "reinforcementLearningActionType": "discrete",
                    "baseModel": "DQN",
                    "epsilon": 0.2,
                    "discountFactor": 0.75,
                    "learningRateNN": 0.01,
                    "maxBatchSize": 10000,
                    "batchSize": 500,
                    "trainingPredictIterationPeriod": -25,
                    "trainingTargetIterationPeriod": -25,
                    "epoch": 150,
                    "calculateTt": 0,
                    "kDefaultAction": [-1],
                    "aDefaultAction": [-1],
                    "earlyStoppingTraining": 0,
                    "earlyStoppingDataSplitTrainingPct": 0.6,
                    "firstHour": 7,
                    "lastHour": 19,
                    "numberDecimalsState": 3,
                    "binaryStateOutputs": 0,
                    "periodsTAStates": [9, 13, 21],
                    "dumpSeconds": 30,
                    "dumpData": 0,
                    "secondsCandles": 56,
                    "otherInstrumentsStates": [],
                    "otherInstrumentsMsPeriods": [],
                    "stopActionOnFilled": 0
                }
            },
            "reporting": {
                "saveTrades": True,
                "savePnL": True,
                "saveOrders": True,
                "saveLogs": True
            },
            "environment": {
                "javaTempDir": "temp/",
                "javaOutputDir": "output/",
                "javaDataDir": "D:\\git repos\\HFTFramework\\data\\"
            }
        }, indent=2)
        # save it into file
        filename = os.getcwd() + os.sep + self.train_input_configuration.get_filename()
        textfile = open(filename, 'w')
        textfile.write(file_content)
        textfile.close()

        command_to_run = self.task + ' "%s" ' % filename
        print('pwd=%s' % os.getcwd())
        if self.VERBOSE_OUTPUT:
            command_to_run += '>%sout.log' % (os.getcwd() + os.sep)
        ret = os.system(command_to_run)
        if ret != 0:
            print("error launching %s" % (command_to_run))

        print('%s finished with code %d' % (self.id, ret))
        # remove input file
        self.state = TrainLauncherState.finished

        if os.path.exists(filename):
            os.remove(filename)


def clean_javacpp():
    from pathlib import Path

    home = str(Path.home())
    java_cpp = home + os.sep + rf'.javacpp\cache'
    print('cleaning java_cpp: %s' % java_cpp)
    os.remove(java_cpp)


def clean_gpu_memory():
    # print(rf"cleaning gpu memory")
    from numba import cuda

    device = cuda.get_current_device()
    device.reset()


class TrainLauncherController:
    def __init__(self, train_launcher: TrainLauncher):
        self.train_launchers = [train_launcher]
        self.max_simultaneous = 1

    def run(self):
        # clean_javacpp()
        sent = []
        start_time = time.time()
        while 1:
            running = 0
            for train_launcher in self.train_launchers:
                if train_launcher.state == TrainLauncherState.running:
                    running += 1
            if (self.max_simultaneous - running) > 0:
                backtest_waiting = [
                    backtest
                    for backtest in self.train_launchers
                    if backtest not in sent
                ]

                for idx in range(
                    min(self.max_simultaneous - running, len(backtest_waiting))
                ):
                    train_launcher = backtest_waiting[idx]
                    print("launching %s" % train_launcher.id)
                    train_launcher.start()
                    sent.append(train_launcher)

            processed = [t for t in sent if t.state == TrainLauncherState.finished]
            if len(processed) == len(self.train_launchers):
                seconds_elapsed = time.time() - start_time
                print(
                    'finished %d training in %d minutes'
                    % (len(self.train_launchers), seconds_elapsed / 60)
                )
                break
            time.sleep(0.01)


if __name__ == '__main__':
    train_input_configuration = TrainInputConfiguration(
        memory_path='memoryReplay_sample.csv',
        output_model_path='output_python.model',
        action_columns=6,
        state_columns=6,
        number_epochs=200,
    )

    train_launcher = TrainLauncher(
        train_input_configuration=train_input_configuration,
        id='main_launcher',
        jar_path=r'd:\git repos\HFTFramework\java\executables\Backtest\target\Backtest.jar',
    )
    train_launcher_controller = TrainLauncherController(train_launcher=train_launcher)
    train_launcher_controller.run()
