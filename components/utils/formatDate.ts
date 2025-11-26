export const formatDate = (date: Date) => {
    // Create options object for date formatting
    const options: Intl.DateTimeFormatOptions = { 
      year: 'numeric', 
      month: 'long', 
      day: 'numeric' 
    };
    
    // Format the date
    return new Intl.DateTimeFormat('en-US', options).format(new Date(date));
  };