<script>
  export let type = null;
  export let data = null;
  export let action_htmls = null;
  export let action_groups = null;
  export let max_width;

  let type_and_data = type + (data === null ? "" : ":" + data.toString());

  $: {
    if (type_and_data.indexOf(":") === -1) {
      type = type_and_data;
      data = null;
    }
    else {
      let type_and_data_split = type_and_data.split(":");
      type = type_and_data_split[0];
      data = parseInt(type_and_data_split[1]);
    }
  }
</script>

<style>
 td {
   padding: 0.25em;
 }

 label {
   white-space: nowrap;
   border: 1px solid gray;
 }
</style>

<td style="max-width: {max_width};">

  <p>
    <label><input type="radio" bind:group={type_and_data} value="v">value function</label>
  </p>

  <p>
    policy logits:<br>
    {#each action_htmls as action_html, action}
      <label>
        <input type="radio" bind:group={type_and_data} value="action:{action}">
        {@html action_html}
      </label>
      <span></span>
    {/each}
  </p>

  <p>
    sums of policy logits:<br>
    {#each action_groups as actions, action_group}
      <label>
        <input type="radio" bind:group={type_and_data} value="action_group:{action_group}">
        {@html actions.map(action => action_htmls[action]).join(" | ")}
      </label>
      <span></span>
    {/each}
  </p>

</td>
