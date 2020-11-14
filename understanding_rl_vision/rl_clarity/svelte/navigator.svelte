<script>
  export let state;
  export let speed = 25;
  export let width;
  export let max_duration;
  export let max_speed = 100;
  export let last_direction = 1;
  export let loop = false;

  const clip_position = function(position) {
    if (loop) {
      position = (position + max_duration + 0.5) % max_duration - 0.5;
    }
    return Math.max(0, Math.min(max_duration - 1, position));
  };
  /* const clip_speed = function(speed) {
   *   return Math.max(0, Math.min(max_speed, speed));
   * }; */

  const update_position = function() {
    let speed_safe = speed;
    if (typeof(speed_safe) !== "number") {
      speed_safe = 0;
    }
    let delta = state.velocity_direction * Math.sign(speed_safe);
    state.position = Math.round(clip_position(state.position + delta));
    let seconds = 1;
    if (speed_safe !== 0) {
      seconds = 1 / Math.abs(speed_safe);
    }
    window.setTimeout(update_position, 1000 * seconds);
  };
  update_position();

  const handle_keydown = function(event) {
    if (event.keyCode === 37 || event.keyCode === 39) {
      let delta = event.keyCode === 37 ? -1 : 1;
      if (state.velocity_direction === 0) {
        state.position = clip_position(state.position + delta);
      }
      else {
        state.velocity_direction = delta;
      }
      last_direction = delta;
      event.preventDefault();
    }
    /* else if (event.keyCode === 38 || event.keyCode === 40) {
     *   speed = clip_speed(speed + (event.keyCode == 38 ? 1 : -1));
     *   event.preventDefault();
     * } */
    else if (event.keyCode === 32) {
      if (state.velocity_direction === 0) {
        state.velocity_direction = last_direction;
      }
      else {
        state.velocity_direction = 0;
      }
      event.preventDefault();
    }
  };
</script>

<style>
 .container {
   display: inline-block;
   position: absolute;
   padding: 0em 0.25em;
 }

 button {
   cursor: pointer;
 }
</style>

<svelte:window on:keydown={handle_keydown}/>

<div style="position: relative;
            width: {width};
            margin: 0.5em auto;
            padding: 0.25em 0em;
            border: 1px solid gray;"
><!-- tabindex="0" -->
  <div class="container" style="left: 0%; text-align: left; z-index: 1;">
    <button
      on:click={() => {state.velocity_direction = 0; state.position = clip_position(state.position - 1); last_direction = -1;}}
    >&#8592;</button>
    <button
      on:click={() => {state.velocity_direction = 0; state.position = clip_position(state.position + 1); last_direction = 1;}}
    >&#8594;</button>
  </div>
  <div class="container" style="left: 0%; right: 0%; text-align: center; z-index: 0;">
    <button on:click={() => state.position = 0}>&#10074;&#9664;</button>
    <button on:click={() => {state.velocity_direction = state.velocity_direction === -1 ? 0 : -1; last_direction = state.velocity_direction;}}>
      {#if state.velocity_direction === -1}&#10074;&#10074;{:else}&#9664;{/if}
    </button>
    <button on:click={() => {state.velocity_direction = state.velocity_direction === 1 ? 0 : 1; last_direction = state.velocity_direction;}}>
      {#if state.velocity_direction === 1}&#10074;&#10074;{:else}&#9658;{/if}
    </button>
    <button on:click={() => state.position = max_duration - 1}>&#9658;&#10074;</button>
  </div>
  <div class="container" style="right: 0%; text-align: right; z-index: 1;">
    <input bind:value={speed} type="number" min="0" max="{max_speed}"> fps
  </div>
  <div style="visibility: hidden;"><button>&nbsp;</button></div>
</div>
