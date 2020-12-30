import { IEdge } from './document';
import { Chart } from './chart';

const validators: {[key: string]: Function} = {};

export function registerValidator(name: string, fun: Function){
  validators[name] = fun;
}

//TODO, need to refactor this to handle dynamcially added module type checks 
function valid(required: any, outputs: any): boolean {
  const keys = Object.keys(required);
  for (let i = 0; i < keys.length; i++) {
    const d = keys[i];
    if (
      !(
        d in outputs &&
        (outputs[d] === null ||
          required[d] === null ||
          required[d] === outputs[d])
      )
    ) {
      return false;
    }
  }
  return true;
}

export function validConnection(that: Chart) {
  return function(from: string, to: string): any {
    if (from === to) {
      return true;
    }
    //canot connect to the port from the same node
    if (from.split('.')[0] === to.split('.')[0]) {
      return false;
    }

    // check whether from is input or output
    const fromOutputPorts = that.outputPorts.has(from);
    if (fromOutputPorts) {
      // to has to be an input port
      if (!that.inputPorts.has(to)) {
        return false;
      }
      //from a output port, multiple connection from the same output port is allowed,
      if (that.props.edges.findIndex((d: IEdge) => d.to === to) >= 0) {
        // if there is already a connection to the input port, it is not valid
        return false;
      }

      const toTypes = that.portTypes[to];
      const fromTypes = that.portTypes[from];
      // if 'any shows up in types, it is valid
      if (
        toTypes.findIndex(d => d[0] === 'any') < 0 &&
        fromTypes.findIndex(d => d[0] === 'any') < 0
      ) {
        const intersection = toTypes.some((x: string[]) => {
          return fromTypes.some((typeNames: string[]) =>
              typeNames.includes(x[0])
           )
          }
        );
        if (! intersection) {
          return false;
        }
      }

      // make sure the requirement is met
      if (from in that.outputMeta && to in that.inputRequriements) {
        let found = false;
        for (let i = 0; i < toTypes.length; i++){
          if (toTypes[i][0] in validators) {
            found = true;
            return validators[toTypes[i][0]](that.inputRequriements[to], that.outputMeta[from]);
            break;
          }
        } 
        if (!found) {
          return valid(that.inputRequriements[to], that.outputMeta[from]);
        }

      } else if (
        !(from in that.outputMeta) &&
        to in that.inputRequriements
      ) {
        return valid(that.inputRequriements[to], {});
      }
    } else {
      //from a input port, only single connection from input port is allowed
      // to has to be output port
      if (!that.outputPorts.has(to)) {
        return false;
      }
      if (that.props.edges.findIndex((d: IEdge) => d.to === from) >= 0) {
        // if there is already a connection to the input port, it is not valid
        return false;
      }
      const toTypes = that.portTypes[to];
      const fromTypes = that.portTypes[from];
      if (
        toTypes.findIndex(d => d[0] === 'any') < 0 &&
        fromTypes.findIndex(d => d[0] === 'any') < 0
      ) {
        const intersection = fromTypes.some((x: string[]) => {
          return toTypes.some((typeNames: string[]) =>
              typeNames.includes(x[0])
           )
          }
        );
        if (! intersection) {
          return false;
        }
      }

      // make sure the requirement is met
      if (to in that.outputMeta && from in that.inputRequriements) {
        let found = false;
        for (let i = 0; i < fromTypes.length; i++){
          if (fromTypes[i][0] in validators) {
            found = true;
            return validators[fromTypes[i][0]](that.inputRequriements[from], that.outputMeta[to]);
            break;
          }
        } 
        if (!found) {
          return valid(that.inputRequriements[from], that.outputMeta[to]);
        }
      } else if (
        !(to in that.outputMeta) &&
        from in that.inputRequriements
      ) {
        return valid(that.inputRequriements[from], {});
      }
    }
    return true;
  };
}
