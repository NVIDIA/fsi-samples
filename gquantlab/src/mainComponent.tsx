import { ReactWidget } from '@jupyterlab/apputils';
import React from 'react';

export class MainView extends ReactWidget {
  constructor() {
    super();
  }

  protected render(): React.ReactElement<any> {
    return <div> this is it </div>;
  }
}
