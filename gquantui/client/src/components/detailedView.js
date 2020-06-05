import React from 'react'

class DetailView extends React.Component {

    handleCloseClick = (event) => {
        if (event) {
            event.preventDefault()
        }
        this.props.close()
    }

    handleLoad = (event) => {
        if (event) {
            event.preventDefault()
        }
        let file = this.props.file.key;
        this.props.handleOpen(file);
    }


    render() {
        let name = this.props.file.key.split('/')
        name = name.length ? name[name.length - 1] : ''

        return (
            <div>
                <h2>Workflow</h2>
                <dl>
                    <dt>Name</dt>
                    <dd>{name}</dd>
                </dl>
                <button onClick={this.handleCloseClick}>Close</button>
                <button onClick={this.handleLoad}>Load</button>
            </div>
        )
    }
}

export default DetailView;