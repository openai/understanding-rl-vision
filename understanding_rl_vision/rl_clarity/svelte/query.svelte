<script>
  export let selected_attribution_id;
  export let video_state;
  export let max_duration;
  export let num_trajectories;

  const get_query = function() {
    let search = window.location.search.substring(1).split("&");
    let query = { };
    for (let s of search) {
      let key_and_value = s.split("=");
      query[key_and_value[0]] = key_and_value[1];
    }
    return query;
  };

  const read_query = function() {
    let query = get_query();
    if (typeof(query.layer) !== "undefined") {
      selected_attribution_id.layer = query.layer;
    }
    if (typeof(query.trajectory) !== "undefined") {
      selected_attribution_id.trajectory = Math.max(0, Math.min(num_trajectories - 1, parseInt(query.trajectory) - 1));
    }
    if (typeof(query.frame) !== "undefined") {
      video_state.position = Math.max(0, Math.min(max_duration - 1, parseInt(query.frame) - 1));
    }
  };

  const construct_query_parts = function(selected_attribution_id, video_state) {
    return [
      ["layer", selected_attribution_id.layer],
      ["trajectory", (selected_attribution_id.trajectory + 1).toString()],
      ["frame", (video_state.position + 1).toString()]
    ];
  };
  let query_parts_init = construct_query_parts(selected_attribution_id, video_state);
  const construct_query_parts_updated = function(selected_attribution_id, video_state) {
    let query_parts = construct_query_parts(selected_attribution_id, video_state);
    let query_parts_updated = [];
    for (let part_number = 0; part_number < query_parts.length; part_number++) {
      if (query_parts[part_number][1] !== query_parts_init[part_number][1]) {
        query_parts_updated.push(query_parts[part_number]);
      }
    }
    return query_parts_updated;
  };

  $: url = (function() {
    let location_parts = [
      window.location.protocol,
      "//",
      window.location.host,
      window.location.pathname,
    ];
    let query_parts_updated = construct_query_parts_updated(selected_attribution_id, video_state);
    if (query_parts_updated.length === 0) {
      return location_parts.join("");
    }
    else {
      let search = query_parts_updated.map(key_and_value => key_and_value[0] + "=" + key_and_value[1]);
      return location_parts.join("") + "?" + search.join("&");
    }
  })();

  const update_url_lockout = 1000;
  let update_url_timeout = null;
  let update_url_prev_time = performance.now() - update_url_lockout - 1;
  let update_url = function(url) {
    if (update_url_timeout !== null) {
      window.clearTimeout(update_url_timeout);
    }
    let curr_time = performance.now();
    if (curr_time - update_url_prev_time > update_url_lockout) {
      window.history.replaceState(null, null, url);
      update_url_timeout = null;
      update_url_prev_time = curr_time;
    }
    else {
      update_url_timeout = window.setTimeout(function (){
        window.history.replaceState(null, null, url);
        update_url_timeout = null;
        update_url_prev_time = performance.now();
      }, update_url_lockout - (curr_time - update_url_prev_time));
    }
  };
  $: {
    update_url(url);
  }

  read_query();
</script>

<svelte:window onpopstate={read_query}/>
