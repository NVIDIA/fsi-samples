from cuquant.dataframe_flow import load_workflow, run
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate the dataframe flow graph')
    parser.add_argument('-t', '--task', help="the yaml task file")
    parser.add_argument('output', help="the output nodes", nargs='+')
    args = parser.parse_args()

    obj = load_workflow(args.task)
    print('output nodes:', args.output)
    run(obj, args.output)

if __name__ == "__main__":
    main()
