import React from 'react';


const TestLogo = () => (
  <img
    src="/images/uic_business.png"
    alt="UIC Business Logo"
    style={{
      width: '200px',      // fix the width
      height: 'auto',      // preserve aspect ratio
      maxHeight: '80px',   // never get taller than this
      objectFit: 'contain',
      display: 'block',    // remove inline-block whitespace if you like
      margin: '10px'
    }}
  />
);

export default TestLogo;
