
function valid(required, outputs){

    let keys = Object.keys(required);

    keys.forEach((d)=>{
        if (!(d in outputs && required[d] === outputs[d])){
            return false;
        }
    });
    return true;
}

export function validConnection(that) {
    return function (from, to) {
        if (from === to){
            return true;
        }
        //canot connect to the port from the same node
        if (from.split(".")[0] === to.split(".")[0]){
            return false;
        }

        // check whether from is input or output
        let fromOutputPorts = that.outputPorts.has(from);
        if (fromOutputPorts){
            // to has to be an input port
            if (!that.inputPorts.has(to)){
                return false;
            };
            //from a output port, multiple connection from the same output port is allowed, 
            if (that.props.edges.findIndex((d) => d.to === to) >= 0){
                // if there is already a connection to the input port, it is not valid
                return false;
            }

            let toTypes = that.portTypes[to];
            let fromTypes = that.portTypes[from];
            let intersection = toTypes.filter(x => fromTypes.includes(x));
            if (intersection.length===0){
                return false;
            }

            // make sure the requirement is met
            if (from in that.outputColumns && to in that.inputRequriements){
                return valid(that.inputRequriements[to], that.outputColumns[from]);
            }
            else if (!(from in that.outputColumns) && to in that.inputRequriements) {
                return valid(that.inputRequriements[to], {});
            }
        }
        else{
            //from a input port, only single connection from input port is allowed
            // to has to be output port
            if (!that.outputPorts.has(to)){
                return false;
            };
            if (that.props.edges.findIndex((d) => d.to === from) >= 0){
                // if there is already a connection to the input port, it is not valid
                return false;
            }
            let toTypes = that.portTypes[to];
            let fromTypes = that.portTypes[from];
            let intersection = toTypes.filter(x => fromTypes.includes(x));
            if (intersection.length===0){
                return false;
            }

            // make sure the requirement is met
            if (to in that.outputColumns && from in that.inputRequriements){
                return valid(that.inputRequriements[from], that.outputColumns[to]);
            }
            else if(!(to in that.outputColumns) && from in that.inputRequriements){
                return valid(that.inputRequriements[from], {});
            }
        }
        return true;
    }
}

