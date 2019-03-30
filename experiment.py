from subprocess import call, STDOUT
import threading

# In this file we run several experiments in the same time

MAX_CORES = 2 # number of processes to run in the same time

'''
    :description - run function  
    :param run_args - the arguments' function
'''


def call_to(run_args):
    print('python main.py {}'.format(run_args))
    param = ['main.py {}'.format(run_args)]
    call(['python', param])

'''
    :description - runs 2 experiments in the same time 
'''


def run_threads():
    run_args = [
                ## NLP data (from 2007)
                # "--nlp \"minimal\" --skip_predict --data_visualization", "--nlp \"minimal\"",
                # "--nlp \"minimal\" --parameter_tuning --enable_vix --windows_mode \"one\" --evaluation_mode \"classifier\"",
                # "--nlp \"minimal\" --parameter_tuning --enable_vix --windows_mode \"one\" --evaluation_mode \"major\"",
                # "--nlp \"minimal\" --parameter_tuning --enable_vix --windows_mode \"one\" --evaluation_mode \"weighted\"",
                # "--nlp \"minimal\" --parameter_tuning --enable_vix --windows_mode \"multiple\" --evaluation_mode \"classifier\"",
                # "--nlp \"minimal\" --parameter_tuning --enable_vix --windows_mode \"multiple\" --evaluation_mode \"major\"",
                # "--nlp \"minimal\" --parameter_tuning --enable_vix --windows_mode \"multiple\" --evaluation_mode \"weighted\"",
                # "--nlp \"minimal\" --parameter_tuning --enable_vix --windows_mode \"sliding\" --evaluation_mode \"classifier\"",
                # "--nlp \"minimal\" --parameter_tuning --enable_vix --windows_mode \"sliding\" --evaluation_mode \"major\"",
                # "--nlp \"minimal\" --parameter_tuning --enable_vix --windows_mode \"sliding\" --evaluation_mode \"weighted\"",

                ## With VIX (from 1993)
                "--skip_predict --data_visualization --enable_vix",
                "--parameter_tuning --enable_vix --windows_mode \"one\" --evaluation_mode \"classifier\"",
                "--parameter_tuning --enable_vix --windows_mode \"one\" --evaluation_mode \"major\"",
                "--parameter_tuning --enable_vix --windows_mode \"one\" --evaluation_mode \"weighted\"",
                "--parameter_tuning --enable_vix --windows_mode \"multiple\" --evaluation_mode \"classifier\"",
                "--parameter_tuning --enable_vix --windows_mode \"multiple\" --evaluation_mode \"major\"",
                "--parameter_tuning --enable_vix --windows_mode \"multiple\" --evaluation_mode \"weighted\"",
                "--parameter_tuning --enable_vix --windows_mode \"sliding\" --evaluation_mode \"classifier\"",
                "--parameter_tuning --enable_vix --windows_mode \"sliding\" --evaluation_mode \"major\"",
                "--parameter_tuning --enable_vix --windows_mode \"sliding\" --evaluation_mode \"weighted\""

                ## Without VIX (all data from 1950)
                # "--skip_predict --data_visualization",
                # "--parameter_tuning --windows_mode \"one\" --evaluation_mode \"classifier\"",
                # "--parameter_tuning --windows_mode \"one\" --evaluation_mode \"major\"",
                # "--parameter_tuning --windows_mode \"one\" --evaluation_mode \"weighted\"",
                # "--parameter_tuning --windows_mode \"multiple\" --evaluation_mode \"classifier\"",
                # "--parameter_tuning --windows_mode \"multiple\" --evaluation_mode \"major\"",
                # "--parameter_tuning --windows_mode \"multiple\" --evaluation_mode \"weighted\"",
                # "--parameter_tuning --windows_mode \"sliding\" --evaluation_mode \"classifier\"",
                # "--parameter_tuning --windows_mode \"sliding\" --evaluation_mode \"major\"",
                # "--parameter_tuning --windows_mode \"sliding\" --evaluation_mode \"weighted\""
                ]
    threads = []
    num_of_threads = 0
    for i in range(0, len(run_args)):
        if num_of_threads >= MAX_CORES:
            for t in threads:
                t.join()
            num_of_threads = 0
        num_of_threads += 1
        t = threading.Thread(target=call_to, args=[run_args[i]])
        threads.append(t)
        t.start()
    for t in threads:
        t.join()


def main():
    run_threads()

if __name__ == '__main__':
    main()
