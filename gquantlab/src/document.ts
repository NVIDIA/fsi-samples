import {
  DocumentWidget,
  DocumentRegistry,
  ABCWidgetFactory
} from '@jupyterlab/docregistry';
import { MainView } from './mainComponent';
import { requestAPI } from './gquantlab';
import YAML from 'yaml';

export class GquantWidget extends DocumentWidget<MainView> {
  constructor(options: DocumentWidget.IOptions<MainView>) {
    super({ ...options });
    this.context = options['context'];
    this.showFile();
  }

  async showFile(): Promise<number> {
    await this.context.ready;
    const yamlContent = this.context.model.toString();
    const objContent = YAML.parse(yamlContent);
    const jsonString = JSON.stringify(objContent);
    this.context.model.contentChanged.connect(this._onContentChanged, this);
    requestAPI<any>('load_graph', {
      body: jsonString,
      method: 'POST'
    })
      .then(data => {
        console.log(data);
      })
      .catch(reason => {
        console.error(
          `The gquantlab server extension appears to be missing.\n${reason}`
        );
      });
    requestAPI<any>('all_nodes')
      .then(data => {
        console.log('all nodes')
        console.log(data);
      })
      .catch(reason => {
        console.error(
          `The gquantlab server extension appears to be missing.\n${reason}`
        );
      });

    return 3;
  }

  private _onContentChanged(): void {
    console.log('content chagned');
  }

  readonly context: DocumentRegistry.Context;
}

/**
 * A widget factory for drawio.
 */
export class GquantFactory extends ABCWidgetFactory<
  GquantWidget,
  DocumentRegistry.IModel
> {
  /**
   * Create a new widget given a context.
   */
  constructor(options: DocumentRegistry.IWidgetFactoryOptions) {
    super(options);
  }

  protected createNewWidget(context: DocumentRegistry.Context): GquantWidget {
    return new GquantWidget({ context, content: new MainView() });
  }
}
