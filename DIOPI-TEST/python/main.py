import subprocess
import argparse
import shlex
import conformance as cf
from conformance.utils import is_ci, error_counter, DiopiException
from conformance.utils import logger
from conformance.model_list import model_list, model_op_list


def parse_args():
    parser = argparse.ArgumentParser(description='Conformance Test for DIOPI')
    parser.add_argument('--mode', type=str, default='test',
                        help='running mode, available options: gen_data, run_test and utest')
    parser.add_argument('--fname', type=str, default='all',
                        help='the name of the function for which the test will run (default: all)')
    parser.add_argument('--model_name', type=str, default='',
                        help='Get op list of given model name')
    parser.add_argument('--filter_dtype', type=str, nargs='*',
                        help='The dtype in filter_dtype will not be processed')
    parser.add_argument('--get_model_list', action='store_true',
                        help='Whether return the supported model list')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.get_model_list:
        print(f"The supported model_list: {model_list}")

        for model_name in model_op_list.keys():
            op_list = model_op_list[model_name]
            print(f"The model {model_name}'s op_list: {op_list}")

        exit(0)

    if args.mode == 'gen_data':
        if args.model_name != '':
            logger.info(f"Now, fname will be disabled, and all {args.model_name}'s ops will be processed.")
        cf.GenInputData.run(args.fname, args.model_name, args.filter_dtype)
        cf.GenOutputData.run(args.fname, args.model_name, args.filter_dtype)
    elif args.mode == 'run_test':
        if args.model_name != '':
            logger.info(f"Now, fname will be disabled, and all {args.model_name}'s ops will be processed.")
        cf.ConformanceTest.run(args.fname, args.model_name, args.filter_dtype)
    elif args.mode == 'utest':
        call = "python3 -m pytest -vx tests"
        subprocess.call(shlex.split(call))  # nosec
    else:
        print("available options for mode: gen_data, run_test and utest")
 
    if is_ci != "null" and error_counter[0] != 0:
        raise DiopiException(str(error_counter[0]) + " errors during this program")
