<?php
ini_set('max_execution_time', 1200);
/* set memory limit to 1000 MB */
//ini_set("memory_limit","1000M");
include "dev-slp.php";

$samAsa_types = array("A1","A2","A3","A4","A5","A6","A7","K1","K2","K3","K4","K5","K6","K7","Km","T1","T2","T3","T4","T5","T6","T7","Tn","Tds","Tdt","Tdu","Tg","Tk","Tp","Tm","Tb","U","Bs2","Bs3","Bs4","Bs5","Bs6","Bs7","Bsd","Bsp","Bsg","Bsmn","Bvp","Bss","Bsu","Bv","Bvs","BvS","BvU","Bb","Di","Ds","E","S","d");
//$alwithweight = array("a:0.019230769230769","A:0.038461538461538","i:0.057692307692308","I:0.076923076923077","u:0.096153846153846","U:0.11538461538462","f:0.13461538461538","F:0.15384615384615","x:0.17307692307692","X:0.19230769230769","e:0.21153846153846","E:0.23076923076923","o:0.25","O:0.26923076923077","k:0.28846153846154","K:0.30769230769231","g:0.32692307692308","G:0.34615384615385","N:0.36538461538462","c:0.38461538461538","C:0.40384615384615","j:0.42307692307692","J:0.44230769230769","Y:0.46153846153846","w:0.48076923076923","W:0.5","q:0.51923076923077","Q:0.53846153846154","R:0.55769230769231","t:0.57692307692308","T:0.59615384615385","d:0.61538461538462","D:0.63461538461538","n:0.65384615384615","p:0.67307692307692","P:0.69230769230769","b:0.71153846153846","B:0.73076923076923","m:0.75","y:0.76923076923077","r:0.78846153846154","l:0.80769230769231","v:0.82692307692308","S:0.84615384615385","z:0.86538461538462","s:0.88461538461538","h:0.90384615384615","M:0.92307692307692","!:0.94230769230769","H:0.96153846153846","-:0.98076923076923");

$dataset1 = tagseparator("input.txt"); // presenting the word:tag format
echo "Culled the words with Tags from input.txt<br/>\n";
echo count($dataset1)."\n";
$dataset2 = multitags($dataset1); // when two tags are applied, changing it to two different entries with different tags.
shuffle($dataset2);
echo count($dataset2)."\n";
$inputset = inputset($dataset2)[0]; // An array of training inputs.
$outputset = inputset($dataset2)[1]; // An array of training inputs.
echo "Created an array of input compounds<br/>\n";
$majorset = majorset($outputset);
$yn = yn($majorset, "T");
echo count($inputset)."\n";
echo count($outputset)."\n";
echo count($majorset)."\n";

$infile = fopen("samAsa_details.txt","w+");
fputs($infile,'inputwords = [');
for ($i=0;$i<count($inputset);$i++)
{
	fputs($infile,'"'.$inputset[$i].'",');
}
fputs($infile,']
');


fputs($infile,'outputwords = [');
for ($i=0;$i<count($outputset);$i++)
{
	fputs($infile,'"'.$outputset[$i].'",');
}
fputs($infile,']
');

fputs($infile,'majorwords = [');
for ($i=0;$i<count($majorset);$i++)
{
	fputs($infile,'"'.$majorset[$i].'",');
}
fputs($infile,']
');

fputs($infile,'yn = [');
for ($i=0;$i<count($yn);$i++)
{
	fputs($infile,'"'.$yn[$i].'",');
}
fputs($infile,']
');

fclose($infile);

/* Create a CSV file too */
$outcsv = fopen('samAsa_details.csv','w+');
$length = count($inputset);
for($i=0;$i<$length;$i++)
{
	fputs($outcsv,$inputset[$i].",".$outputset[$i].",".$majorset[$i]."\n");
}
fclose($outcsv);

/* Functions used in the code */
function multitags($dataset1)
{
	foreach ($dataset1 as $value)
	{
		if (strpos($value,",")!==false )
		{
			$part = explode(":",$value);
			$tagpart = explode(",",$part[1]);
			foreach ($tagpart as $tag)
			{
				$val[] = convert1($part[0]).":".$tag;
			}
		}
		else
		{
			$val[] = convert1($value);
		}
	}
	return $val;
}

function tagseparator($filename)
{
	global $samAsa_types;
	$lines = file($filename);
	$counter1 = 0; $counter2 = 0;
	foreach ($lines as $line)
	{
		$words = explode(" ",$line);
		foreach ($words as $word)
		{
			if (preg_match('/^[<]([^-]*)[-]([^>]*)[\>]/',$word) && !strpos($word,"Ds-") && strlen($word)<=50)
			{
				$word = trim($word);
				$word = trim($word,'?');
				$word = str_replace(array('_','+','.'),array('',' ',''),$word); // For input of PV
				$word = str_replace(array('  '),array(' '),$word); // For input of PV
				$word = str_replace(array('{2}','{3}'),array('',''),$word); // For input of RSVP
				if (substr_count($word,"<")===1)
				{
				$counter1++;
				$sep = explode('>',$word);
				$sep[0] = trim($sep[0],"<");
					if (in_array($sep[1],$samAsa_types)) // Removing the erraneous entries e.g. without space etc.
					{
						$twowords[]=$sep[0].":".$sep[1];						
					}
				}
				else // right now not accounting for them.
				{
				$counter2++;
				$multiwords[]=$word;
				}
			}
		}
	}
	return $twowords;
}

function weightage()
{
	$alphabets = array("a","A","i","I","u","U","f","F","x","X","e","E","o","O","k","K","g","G","N","c","C","j","J","Y","w","W","q","Q","R","t","T","d","D","n","p","P","b","B","m","y","r","l","v","S","z","s","h","M","!","H","-");
	$i = 1;
	foreach ($alphabets as $value)
	{
		$val1[] = $value;
		$val2[] = $i/52;
		//echo $value.":".($i/52).'","'; // To generate $alwithweight
		$i++;
	}
}
function inputset($dataset2)
{
	global $samAsa_types;
	for ($i=0;$i<count($dataset2);$i++)
	{
		$parts = explode(':',$dataset2[$i]);
		$parts = array_map('trim',$parts);
		if (in_array($parts[1],$samAsa_types) && preg_match('/^([a-zA-Z-]*)$/',$parts[0]) )
		{
			$parts[0] = str_replace(array("1","2","3","4","5","6","7","8","9","0","{","}","."),array("","","","","","","","","","","","",""),$parts[0]);
			$input[] = $parts[0];
			$output[] = $parts[1];
		}
	}
	$val[0] = $input;
	$val[1] = $output;
	return $val;
}

function majorset($outputset)
{
	$fulllist = array("A1","A2","A3","A4","A5","A6","A7","K1","K2","K3","K4","K5","K6","K7","Km","T1","T2","T3","T4","T5","T6","T7","Tn","Tds","Tdt","Tdu","Tg","Tk","Tp","Tm","Tb","Bs2","Bs3","Bs4","Bs5","Bs6","Bs7","Bsd","Bsp","Bsg","Bsmn","Bvp","Bss","Bsu","Bvs","BvS","Bv","BvU","Bb","Di","Ds","E","S","d","U","BT");
    $majorlist = array("A","A","A","A","A","A","A","K","K","K","K","K","K","K","K","T","T","T","T","T","T","T","T","T","T","T","T","T","T","T","T","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","B","D","D","D","D","D","T","B");

	foreach ($outputset as $value)
	{
		$val[] = str_replace($fulllist,$majorlist,$value);	
	}
	return $val;
}
function yn($majorset, $classname)
{
	foreach ($majorset as $value)
	{
		if ($value === $classname)
		{
			$val[] = str_replace($value,"Y",$value);			
		}
		else
		{
			$val[] = str_replace($value,"N",$value);
		}
	}
	return $val;
}

?>
