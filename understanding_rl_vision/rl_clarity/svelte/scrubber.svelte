<script>
  export let state;
  export let visible_duration;
  export let max_duration;

  const clip_position = function(position) {
    return Math.max(0, Math.min(max_duration - 1, position));
  };

  let drag_div;
  let drag_position = null;
  const drag_start = function(event) {
    if (event.buttons === undefined ? event.which === 1 : event.buttons === 1) {
      state.velocity_direction = 0;
      let delta = (event.pageX / drag_div.scrollWidth) * visible_duration;
      drag_position = state.position + delta;
      event.preventDefault();
    }
  };
  const drag_update = function(event) {
    if (drag_position !== null) {
      let delta = (event.pageX / drag_div.scrollWidth) * visible_duration;
      state.position = Math.round(clip_position(drag_position - delta));
    }
  };
  const drag_end = function() {
    drag_position = null;
  };
</script>

<style>
 .scrubber {
   height: 100%;
   width: 100%;
 }

 .indicator {
   height: 100%;
   width: 2px;
   margin: 0 auto;
   border-width: 0px 6px;
   border-color: black;
   border-style: solid;
   opacity: 0.4;
 }
</style>

<svelte:window on:mousemove={drag_update} on:mouseup={drag_end}/>

<div
  bind:this={drag_div}
  class="scrubber"
  style="cursor: {drag_position === null ? 'grab' : 'grabbing'}"
  on:mousedown={drag_start}
>
  <div
    style="z-index: -1;
           position: absolute;
           height: 100%;
           width: 100%;"
  ></div>
  <div class="indicator"></div>
</div>
