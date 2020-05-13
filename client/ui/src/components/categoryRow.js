import React from 'react';

class CategoryRow extends React.Component {
    render() {
      const category = this.props.category;
      return (
        <tr>
          <th colSpan="1">
            {category}
          </th>
        </tr>
      );
    }
  }

export default CategoryRow;