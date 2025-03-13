document.addEventListener('DOMContentLoaded', function() {
    const array = new Uint32Array(1);
    window.crypto.getRandomValues(array);
    const userId = "id" + array[0].toString(16);
    const textarea = document.getElementById('prompt');
    const recom = document.getElementById('recommendation');
    
    let activityData = [];

    // waits for the page to fully load before attaching event listeners
    function logActivity(event) {
      //if (event.target.parentNode.id == "recommendation") console.log(event.target.textContent);
      // function records the data and logs it to the console
      const timestamp = new Date().toISOString();
      const buttonId = event.target.getAttribute('bx--tag');
      const activity = {
        //records event type, target element, timestamp for a userid
        userId: userId,
        type: event.type,
        target: event.target.tagName,
        timestamp: timestamp,
        text: textarea.value,
        recommentation: recom.textContent,
        label: event.target.textContent
      };
      activityData.push(activity);
      // logs to console
      console.log(JSON.stringify(activity)); 
    }
  
    ['click', 'keypress'].forEach(eventType => {
      document.addEventListener(eventType, logActivity, true); 
    });
  
    window.addEventListener('beforeunload', function() {
        // send data to server before page unload or when user navigates away or closes the page
        //  with navigator.sendBeacon
        if (activityData.length > 0) {
            localStorage.setItem('analyticsData', JSON.stringify(activityData));
            var blob = new Blob([JSON.stringify({activityData})], {type : 'application/json; charset=UTF-8'});
            navigator.sendBeacon('/log', blob);
        }
    });
 });
