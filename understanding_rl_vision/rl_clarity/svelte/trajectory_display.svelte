<script>
  export let actions = null;
  export let rewards = null;
  export let dones = null;
  export let policy_logits = null;
  export let state;
  export let action_htmls = null;
  export let action_colors = null;
  export let bold_color = null;
  export let policy_display_height;
  export let policy_display_width;

  $: policy_probs = (function() {
    let probs = policy_logits[state.position].map(Math.exp);
    let sum_probs = probs.reduce((total, prob) => total + prob, 0);
    probs = probs.map(prob => prob / sum_probs);
    return probs;
  })();
  $: policy_cum_probs = (function() {
    let cum_probs = policy_probs.reduce((cum_probs, prob) =>
      cum_probs.concat([cum_probs[cum_probs.length - 1] + prob]), [0]);
    cum_probs.pop();
    return cum_probs;
  })();
</script>

<style>
 .container {
   position: absolute;
   overflow: hidden;
   white-space: nowrap;
 }

 .label {
   font-weight: bold;
   color: white;
 }

 .first-row td {
   width: 20%;
   white-space: nowrap;
   text-align: center;
 }

 td {
   border: 1px solid gray;
 }
</style>

<table style="margin: 0 auto; border-collapse: collapse;">
  <tr style="height: {policy_display_height};" class="first-row">
    <td>
      {#if state.position > 0 && rewards[state.position - 1] !== 0}
        last reward:
        <span style="font-weight: bold; color: {bold_color};">{rewards[state.position - 1]}</span>
      {/if}
    </td>
    <td>
      {#if state.position > 0 && dones[state.position - 1]}
        <span style="font-weight: bold; color: {bold_color};">new episode</span>
      {/if}
    </td>
    <td>frame: <span style="font-weight: bold; color: {bold_color};">{state.position + 1}</span></td>
    <td style="border-bottom: 1px double white;">policy:</td>
    <td>
      next action:
      <span style="font-weight: bold; color: {bold_color};">
        {@html action_htmls[actions[state.position]]}
      </span>
    </td>
  </tr>
  <tr>
    <td colspan="5">
      <div style="position: relative;
                  height: {policy_display_height};
                  width: {policy_display_width};
                  margin: 0 auto;"
      >
        {#each policy_probs as prob, action}
          <div
            class="container label center-text"
            style="left: {(policy_cum_probs[action] * 100).toFixed(10)}%;
                   height: 100%;
                   width: {(prob * 100).toFixed(10)}%;
                   background-color: {action_colors[action]};"
          >{@html action_htmls[action]}</div>
        {/each}
      </div>
    </td>
  </tr>
</table>
