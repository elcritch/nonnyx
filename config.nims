if defined(macosx):
  switch("passL", "-L/opt/homebrew/lib -Wl,-rpath,/opt/homebrew/lib/")
