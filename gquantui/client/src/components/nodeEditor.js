import React from 'react';
import styled from '@emotion/styled';
import Form from '@rjsf/core';

class NodeEditor extends React.Component {
     constructor(props) {
      super(props);
      this.handleSave = this.handleSave.bind(this);
      this.handleDelete = this.handleDelete.bind(this);
      this.myRef = React.createRef();
    }

    handleSave(d){
      let id = this.props.nodeDatum.id;
      let newNodes = this.props.nodes.filter((d)=>d.id !== id);
      let nodeName = this.myRef.current.value;
      //let nodeConf = d.currentTarget.parentElement.getElementsByTagName('textarea')[0].value;
      if (newNodes.findIndex((d)=>d.id === nodeName) >= 0){
        window.alert(`the node id ${nodeName} is not unique`);
        return;
      }
      this.props.nodeDatum.id = nodeName;
      this.props.nodeDatum.conf = d.formData;
      let newEdges = this.props.edges.map((d)=>{
        let oldNodeName = d.from.split(".")[0];
        if (oldNodeName===id){
          d.from = nodeName+"."+d.from.split(".")[1];
        } 
        oldNodeName = d.to.split(".")[0];
        if (oldNodeName===id){
          d.to = nodeName+"."+d.to.split(".")[1];
        } 
        return d;
      })
      this.props.setChartState({"nodes":this.props.nodes, 'edges': newEdges});
      this.props.setMenuState({'opacity': 0, x: -1000, y:-1000});
    }

    handleDelete(){
        let id = this.props.nodeDatum.id;
        let newNodes = this.props.nodes.filter((d)=>d.id !== id);
        let newEdges = this.props.edges.filter((d)=>{
            return (d.from.split(".")[0]!==id && d.to.split(".")[0]!==id);
        });
        this.props.setChartState({"nodes":newNodes, "edges":newEdges});
        this.props.setMenuState({'opacity': 0, x: -1000, y:-1000});
    }
 
    render() {
        let x = (this.props.x + 25) + 'px';
        let y = (this.props.y) + 'px';

        const Editor = styled.div`
            text-align: left;
            padding: 10px;
            position: absolute;
            background-color: yellowgreen;
            opacity: ${this.props.opacity};
            left: ${x};
            top:  ${y};
        `
        const Button = styled.button`
          background-color: red;
        `
        console.log(this.props.nodeDatum);

            //<TextArea placeholder="configuration json" defaultValue={JSON.stringify(this.props.nodeDatum.conf, null, 2)}/>
      return (
        <Editor>
          <div>
            <span>Node id:</span><input type="text" placeholder="unique node name" defaultValue={this.props.nodeDatum.id} ref={this.myRef}/>
        </div>
            <Form schema={this.props.nodeDatum.schema} formData={this.props.nodeDatum.conf} uiSchema={this.props.nodeDatum.ui} onSubmit={this.handleSave} />
            <Button onClick={this.handleDelete}>Delete</Button>
        </Editor>
      );
    }
  }
  
export default NodeEditor;