## Use VSCode to do development

### Lanuch docker container for development

Install the [VS Code Remote Development addon](https://code.visualstudio.com/docs/remote/remote-overview).
Setup the system environment variable `GQUANT_ROOT` which points to the gquant root directory in the local/remote machine that has the gQuant image compiled. 
Either in the `settings.json` or `.code-workspace` file, set the `docker.host` variable pointing to the machine which has gQuant docker image. E.g.
```
	"docker.host": "ssh://username@remote.host.ip"
```
Launch the container by `Remote-Containers: Open Workspace in Container` and select the `workspace.code-workspace` file in the `gquantui directory`. 
VSCode will prompt that a `.code-workspace` in side the workspace, select to open it. You can see two sub-projects open in the workspace.


### Install Rreact Javascript dependences 
Open a terminal inside the container for `client` project. Run following commands to install all the node modules for the web client.

```bash
source activate rapids
npm install
```

### Start the server 
Open a terminal inside the container for `server` project. Click the `RUN Python: Flask` button in VS Code to start the REST server.

### Start the client side development server 
Open a terminal inside the container for `client` project. Run following command to open the client development server.

```bash
npm start
```

Click the `Launch Chrome` button in VS Code to start the chrome web browser locally. It will start the App inside the browser.


## Run the App in production

### Build React client to production  

Open a terminal inside the container for `client` project. Run following command to build the client code for production

```bash
npm run build
```

Builds the app for production to the `build` folder.

It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes. To test it, we can use Flask to host it.

Open a terminal inside the container for `server` project. Run following command to host it:

```bash
bash start.sh
```

The hosted Web App can be accessed at following URL:

```
http://localhost:8888/index.html
```
