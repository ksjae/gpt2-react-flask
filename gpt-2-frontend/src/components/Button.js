import { Button as MaterialButton } from '@material-ui/core';
import React from 'react';


const Button = ({ onClick }) => (
    <MaterialButton
        style={{ marginTop: '1em', width: 'fit-content' }}
        onClick={onClick}
        variant="outlined"
        color="primary">
        말 만들어내기
  </MaterialButton>
)

export default Button;