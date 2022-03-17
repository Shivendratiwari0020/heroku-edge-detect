import React from 'react';
import "./datatable.css"  


export default function Datatable({ data }) {
  const columns = data[0] && Object.keys(data[0]);
  return (
    <table cellPadding={5} cellSpacing={5}>
      <thead>
        <tr>
          {data[0] && columns.map((heading) => <th>{heading}</th>)}
        </tr>
      </thead>
      <tbody>
        {data.map((row) => (
          <tr>
            {columns.map((column) => (
              <td>{row[column]}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}