from greenflow.dataframe_flow import TaskGraph
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate the dataframe flow graph')
    parser.add_argument('-t', '--task', help="the yaml task file")
    parser.add_argument('output', help="the output nodes", nargs='+')
    args = parser.parse_args()
    import pudb
    pudb.set_trace()

    task_graph = TaskGraph.load_taskgraph(args.task)
    print('output nodes:', args.output)
    task_graph.run(args.output)


if __name__ == "__main__":
    main()
