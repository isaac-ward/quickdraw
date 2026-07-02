Ideas of the user, do not incorporate into designs. Ignore. Just for the user to think about stuff

can we add another thing to the
  manfiold/ logs? its that horizon thing but as an animation. so it starts at h01, then interpolate to h02,
  h04, 8, 16, 32, 64 to show how accurate the denoising is over longer predictions. just in the dataspace over
  the position coordinates. hold on each timestep poitn cloud for 2 second, then tween (slow, fast, slow)
  between the next point cloud in 1 second. if a frmae is gonna eb identical don't rerender it. just shows how
  good the predictions are over long horizons. actually let me get some feedback. should this be in the
  eval_ood_horizon? and then we can keep doubling until we're quite far? we already would have generated the
  data right? thoughts? how many long horizon eval trajectories are there?



   also where are the image head renders in eval ood horizon? and why did you remove the progress of the
  eval prcedures in progress.log? why is there ood_horizon-mm and not the old eval_ood_horizon in the
  loggingg? it seems like you tried to reimplement mms tuff from scratch rather than generalize and extend
  the existing logging. THis was a huge mistake (progress.log updates should be saying '10% done', '20%
  done', etc. for each of the eval procedures (eval_control, eval diffusiin,eval_ood_horizon) and for each
  of the main output logging products WITHIN in eval procedure (i.e. the predicteed image/gt video, and the
  scene animation in eval_horizon). also whre is the image head stuff that we discussed in the
  eval_ood_horizon. I think eval_control is hung also - look at progress.log and get back to me. i think we
  should kill everything and launch a smokle run when ready that has an eval at epoch 1 (not zero) so we
  can confirm everything works and is not missing. the smoke would be a non image diffusion run on gpu 0,
  and an image diffusion run on gpu 1. ALSO in general there SHOULDN'T BE A VECTOR PATH AND AN MM PATH.
  Just one general path that can be aprameterized to do image only, proprio only, or both. Does that make
  sense? So for example you started this fix by making mm open loop reporting report the same 4 m,etrics as
  proprio only open loop. But there should not be thse two things in the codebase - just ONE! gett it? as

  glad you're checking out the error that caused the crash, but the refactor checklist should be top of
  mind before launching another test. look at my last prompt and build that tcheclikst and start addressing
  the requests one by one so thatyoudon't lose track. be very granular when you turn what i requested into
  checklist items



  gifs